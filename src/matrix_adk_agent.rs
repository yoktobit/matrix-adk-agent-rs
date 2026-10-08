use std::{
    collections::{HashMap, HashSet},
    sync::atomic::{AtomicU64, Ordering},
    sync::{Arc, Mutex},
    time::Duration,
};

use anyhow::Context;
use matrix_sdk::ruma::events::Mentions;
use matrix_sdk::ruma::events::SyncMessageLikeEvent;
use matrix_sdk::ruma::events::room::member::StrippedRoomMemberEvent;
use matrix_sdk::ruma::events::room::message::{MessageType, RoomMessageEventContent};
use matrix_sdk::ruma::{OwnedUserId, UserId};
use matrix_sdk::{Client, Room, RoomMemberships, RoomState};
use regex::Regex;

use crate::adk::AdkOpenAiAgent;
use crate::matrix::MatrixAgent;

#[derive(Clone)]
pub struct MatrixAdkAgent {
    matrix_agent: Arc<MatrixAgent>,
    adk_agent: Arc<AdkOpenAiAgent>,
    auto_join: bool,
    auto_join_inviter_regex: Option<Regex>,
    task_counter: Arc<AtomicU64>,
    task_requesters: Arc<Mutex<HashMap<String, String>>>,
    pending_helper_tasks: Arc<Mutex<HashMap<String, PendingHelperTask>>>,
    // TODO: Weitere Handler erlauben
    // join_handlers: Vec<JoinHandler>,
    // message_handlers: Vec<MessageHandler>,
}

fn inviter_matches(regex: Option<&Regex>, inviter: &str) -> bool {
    regex.is_none_or(|regex| regex.is_match(inviter))
}

impl MatrixAdkAgent {
    pub fn new(
        matrix_agent: MatrixAgent,
        adk_agent: AdkOpenAiAgent,
        auto_join: bool,
        // join_handlers: Vec<JoinHandler>,
        // message_handlers: Vec<MessageHandler>,
    ) -> anyhow::Result<Self> {
        let auto_join_inviter_regex = std::env::var("MATRIX_AUTOJOIN_INVITER_REGEX")
            .ok()
            .filter(|pattern| !pattern.trim().is_empty())
            .map(|pattern| {
                Regex::new(&pattern)
                    .with_context(|| format!("invalid MATRIX_AUTOJOIN_INVITER_REGEX: {pattern}"))
            })
            .transpose()?;

        let matrix_adk_agent = Self {
            matrix_agent: Arc::new(matrix_agent),
            adk_agent: Arc::new(adk_agent),
            auto_join: auto_join,
            auto_join_inviter_regex,
            task_counter: Arc::new(AtomicU64::new(1)),
            task_requesters: Arc::new(Mutex::new(HashMap::new())),
            pending_helper_tasks: Arc::new(Mutex::new(HashMap::new())),
            // join_handlers: join_handlers,
            // message_handlers: message_handlers,
        };
        Ok(matrix_adk_agent)
    }

    pub async fn connect_matrix(self: &Arc<Self>) -> Result<(), anyhow::Error> {
        self.matrix_agent().connect_matrix().await?;
        if self.auto_join {
            let agent = Arc::clone(self);
            self.matrix_agent.client().add_event_handler(
                move |room_member: StrippedRoomMemberEvent, client: Client, room: Room| {
                    let agent = Arc::clone(&agent);
                    async move {
                        agent
                            .on_stripped_state_member(room_member, client, room)
                            .await
                    }
                },
            );
        }
        Ok(())
    }

    pub async fn run(self: &Arc<Self>) -> Result<(), anyhow::Error> {
        self.add_default_message_handler();
        self.matrix_agent.sync().await
    }

    fn matrix_agent(&self) -> Arc<MatrixAgent> {
        self.matrix_agent.clone()
    }

    pub fn add_default_message_handler(self: &Arc<Self>) {
        let agent = Arc::clone(self);
        self.matrix_agent
            .client
            .add_event_handler(|event, room| async move {
                agent.on_room_message(event, room).await;
            });
    }

    async fn on_stripped_state_member(
        self: &Arc<Self>,
        room_member: StrippedRoomMemberEvent,
        client: Client,
        room: Room,
    ) {
        if room_member.state_key != client.user_id().unwrap() {
            return;
        }

        let inviter = room_member.sender.to_string();
        if !inviter_matches(self.auto_join_inviter_regex.as_ref(), &inviter) {
            println!(
                "Ignoring invitation from {inviter} for room {}",
                room.room_id()
            );
            return;
        }

        tokio::spawn(async move {
            println!("Autojoining room {}", room.room_id());
            let mut delay = 2;

            while let Err(err) = room.join().await {
                // retry autojoin due to synapse sending invites, before the
                // invited user can join for more information see
                // https://github.com/matrix-org/synapse/issues/4345
                eprintln!(
                    "Failed to join room {} ({err:?}), retrying in {delay}s",
                    room.room_id()
                );

                matrix_sdk::sleep::sleep(Duration::from_secs(delay)).await;
                delay *= 2;

                if delay > 3600 {
                    eprintln!("Can't join room {} ({err:?})", room.room_id());
                    break;
                }
            }
            if delay <= 3600 {
                println!("Successfully joined room {}", room.room_id());
            }
        });
    }

    pub async fn on_room_message(
        &self,
        event: SyncMessageLikeEvent<RoomMessageEventContent>,
        room: Room,
    ) {
        if room.state() != RoomState::Joined {
            return;
        }

        match event {
            SyncMessageLikeEvent::Original(event) => {
                let MessageType::Text(text_content) = event.content.msgtype else {
                    return;
                };

                let message_body = text_content.body.clone();
                if Self::starts_with_task_failed_marker(&message_body) {
                    println!("got failed-task message, ignoring");
                    return;
                }

                let sender = event.sender.to_string();
                let room_id = room.room_id().to_string();
                let own_user_id = room.own_user_id().to_owned();
                let mut mentioned = false;
                let mut incoming_mentioned_user_ids: HashSet<OwnedUserId> = HashSet::new();
                for mention in event.content.mentions.iter() {
                    if mention.room {
                        mentioned = true;
                    }
                    for mentioned_user_id in &mention.user_ids {
                        if mentioned_user_id == &own_user_id {
                            mentioned = true;
                        } else {
                            incoming_mentioned_user_ids.insert(mentioned_user_id.clone());
                        }
                    }
                }

                let observed_tag = if mentioned {
                    "[ObservedMentionedMessage]"
                } else {
                    "[ObservedUnmentionedMessage]"
                };
                let observed_message = format!(
                    "{observed_tag}\nSender: {sender}\nRoom: {room_id}\nMessage: {message_body}"
                );
                if let Err(err) = self
                    .adk_agent
                    .record_observed_message(room_id.clone(), observed_message)
                    .await
                {
                    eprintln!("failed to record observed message: {err}");
                }

                if let Some(introduction) = Self::extract_introduction(&message_body) {
                    if sender != room.own_user_id().to_string() {
                        self.adk_agent
                            .remember_introduction(sender.clone(), introduction.clone());
                        println!("Stored introduction from {sender}: {introduction}");
                    }
                }

                if !mentioned {
                    println!("message recorded to session; not mentioned, skipping response");
                    return;
                }

                let task_context = self.build_task_context(&sender, room_id, &message_body);
                let Some(question) =
                    self.build_question_with_helper_barrier(&sender, &task_context)
                else {
                    return;
                };
                println!("got message and was mentioned, asking llm: {question}");
                if let Err(_) = room.typing_notice(true).await {}
                let result_content = match self.adk_agent.ask(room.room_id().into(), question).await
                {
                    Ok(result_content) => result_content,
                    Err(err) => {
                        eprintln!("failed to ask llm: {err:#}");
                        let _ = room.typing_notice(false).await;
                        return;
                    }
                };
                println!("Result: {result_content}");

                if let Err(_) = room.typing_notice(false).await {}

                if !result_content.trim().is_empty() {
                    let response = Self::parse_agent_response(&result_content);
                    let effective_task_id = response.effective_task_id(&task_context);
                    let mut mention_targets =
                        self.resolve_mentions(&response, &task_context, &room).await;
                    for incoming_user_id in incoming_mentioned_user_ids {
                        if self
                            .adk_agent
                            .is_known_helper_user_id(incoming_user_id.as_ref())
                        {
                            continue;
                        }
                        let mut label = incoming_user_id.to_string();
                        if let Some(room_label) =
                            Self::lookup_room_label(&room, incoming_user_id.as_ref()).await
                        {
                            label = room_label;
                        }
                        mention_targets.push(MentionTarget {
                            user_id: incoming_user_id,
                            label,
                            source_tag: None,
                        });
                    }
                    mention_targets.sort_by(|left, right| left.user_id.cmp(&right.user_id));
                    mention_targets.dedup_by(|left, right| left.user_id == right.user_id);
                    self.update_pending_helper_task_state(
                        &effective_task_id,
                        &task_context.requester,
                        &response,
                        &mention_targets,
                    );
                    let outgoing_text = self.compose_outgoing_message(&response, &task_context);
                    let (outgoing_text, outgoing_html) =
                        Self::attach_visible_mentions(outgoing_text, &mention_targets);
                    let mentioned_user_ids = mention_targets
                        .iter()
                        .map(|target| target.user_id.clone())
                        .collect::<Vec<_>>();

                    let mut content = if let Some(html) = outgoing_html {
                        RoomMessageEventContent::text_html(outgoing_text, html)
                    } else {
                        RoomMessageEventContent::text_plain(outgoing_text)
                    };
                    if !mentioned_user_ids.is_empty() {
                        content = content.add_mentions(Mentions::with_user_ids(mentioned_user_ids));
                    }

                    println!("sending");
                    room.send(content).await.unwrap();
                    println!("message sent");
                }
            }
            SyncMessageLikeEvent::Redacted(_redacted) => {}
        }
    }

    fn extract_introduction(message: &str) -> Option<String> {
        let trimmed = message.trim_start();
        let introduction = trimmed
            .strip_prefix("[Introduction]")
            .or_else(|| trimmed.strip_prefix("[introduction]"))?
            .trim_start_matches(':')
            .trim();
        let introduction = Self::strip_helper_tags(introduction);
        let introduction = introduction.trim();

        if introduction.is_empty() {
            return None;
        }

        Some(introduction.to_string())
    }

    fn strip_helper_tags(message: &str) -> String {
        let mut out = String::with_capacity(message.len());
        let mut remaining = message;
        while let Some(start) = remaining.find("@[") {
            let after = &remaining[start + 2..];
            let Some(end) = after.find(']') else {
                break;
            };
            out.push_str(&remaining[..start]);
            remaining = &after[end + 1..];
        }
        out.push_str(remaining);
        out
    }

    fn starts_with_task_failed_marker(message: &str) -> bool {
        let mut trimmed = message.trim_start();
        if trimmed.starts_with("[Task:") {
            if let Some(end) = trimmed.find(']') {
                trimmed = trimmed[end + 1..].trim_start();
            }
        }
        if !trimmed.starts_with("[TaskFailed:") {
            return false;
        }

        Self::extract_marker_value(trimmed, "[TaskFailed:", ']').is_some()
    }

    fn extract_helper_tags(message: &str) -> Vec<String> {
        let mut helper_names = Vec::new();
        let mut remaining = message;

        while let Some(start) = remaining.find("@[") {
            let after_start = &remaining[start + 2..];
            let Some(end) = after_start.find(']') else {
                return helper_names;
            };

            let helper_name = after_start[..end].trim();
            if !helper_name.is_empty() {
                helper_names.push(helper_name.to_string());
            }

            remaining = &after_start[end + 1..];
        }

        for helper_name in Self::extract_bare_helper_tags(message) {
            if !helper_names.contains(&helper_name) {
                helper_names.push(helper_name);
            }
        }

        helper_names
    }

    fn extract_bare_helper_tags(message: &str) -> Vec<String> {
        let chars = message.char_indices().collect::<Vec<_>>();
        let mut helper_names = Vec::new();
        let mut index = 0;

        while index < chars.len() {
            let (start_offset, current) = chars[index];
            if current != '@' {
                index += 1;
                continue;
            }

            if chars.get(index + 1).is_some_and(|(_, next)| *next == '[') {
                index += 1;
                continue;
            }

            if index > 0 {
                let previous = chars[index - 1].1;
                if previous.is_alphanumeric() || matches!(previous, '_' | '.' | '+' | '-') {
                    index += 1;
                    continue;
                }
            }

            let mut end_offset = start_offset + current.len_utf8();
            let mut cursor = index + 1;
            while let Some((offset, ch)) = chars.get(cursor).copied() {
                if ch.is_alphanumeric() || matches!(ch, '_' | '-' | '.' | ':') {
                    end_offset = offset + ch.len_utf8();
                    cursor += 1;
                    continue;
                }
                break;
            }

            if end_offset == start_offset + current.len_utf8() {
                index += 1;
                continue;
            }

            let helper_name = message[start_offset + current.len_utf8()..end_offset]
                .trim()
                .trim_end_matches(|ch: char| matches!(ch, '.' | ',' | '!' | '?' | ';' | ')'));
            if !helper_name.is_empty() {
                helper_names.push(helper_name.to_string());
            }

            index = cursor;
        }

        helper_names
    }

    fn build_task_context(&self, sender: &str, room_id: String, message: &str) -> TaskContext {
        let parsed = TaskMetadata::parse(message);
        let task_id = parsed.task_id.unwrap_or_else(|| self.next_task_id());
        let requester =
            self.resolve_requester_for_task(&task_id, parsed.requester.as_deref(), sender);

        TaskContext {
            task_id,
            requester,
            sender: sender.to_string(),
            room_id,
            original_message: parsed.body,
            has_existing_task_id: parsed.had_task_id,
        }
    }

    fn next_task_id(&self) -> String {
        let task_number = self.task_counter.fetch_add(1, Ordering::Relaxed);
        format!("task-{task_number}")
    }

    fn resolve_requester_for_task(
        &self,
        task_id: &str,
        explicit_requester: Option<&str>,
        sender: &str,
    ) -> String {
        let mut task_requesters = match self.task_requesters.lock() {
            Ok(task_requesters) => task_requesters,
            Err(poisoned) => poisoned.into_inner(),
        };

        Self::resolve_requester_for_task_map(
            &mut task_requesters,
            task_id,
            explicit_requester,
            sender,
        )
    }

    fn resolve_requester_for_task_map(
        task_requesters: &mut HashMap<String, String>,
        task_id: &str,
        explicit_requester: Option<&str>,
        sender: &str,
    ) -> String {
        if let Some(requester) = explicit_requester
            .map(str::trim)
            .filter(|requester| !requester.is_empty())
        {
            let requester = requester.to_string();
            task_requesters.insert(task_id.to_string(), requester.clone());
            return requester;
        }

        if let Some(requester) = task_requesters.get(task_id) {
            return requester.clone();
        }

        let requester = sender.to_string();
        task_requesters.insert(task_id.to_string(), requester.clone());
        requester
    }

    fn parse_agent_response(message: &str) -> AgentResponse {
        let completion_task_id = Self::extract_marker_value(message, "[TaskComplete:", ']');
        let requester = Self::extract_marker_value(message, "[Requester:", ']');
        let task_id = Self::extract_marker_value(message, "[Task:", ']');
        let without_internal_markers = Self::strip_marker(message, "[TaskComplete:");
        let without_requester_marker = Self::strip_marker(&without_internal_markers, "[Requester:");
        let text = Self::strip_marker(&without_requester_marker, "[Task:");
        let helper_names = Self::extract_helper_tags(&text);

        AgentResponse {
            text: text.trim().to_string(),
            helper_names,
            task_id,
            completion_task_id,
            requester,
        }
    }

    fn compose_outgoing_message(
        &self,
        response: &AgentResponse,
        task_context: &TaskContext,
    ) -> String {
        let is_completion = response.is_completion();
        let mut text = response.text.clone();
        let effective_task_id = response.effective_task_id(task_context);
        if Self::starts_with_task_failed_marker(&text) {
            return text;
        }
        let should_prefix_task = !response.helper_names.is_empty()
            || response.completion_task_id.is_some()
            || task_context.has_existing_task_id;

        if should_prefix_task {
            text = format!("[Task: {effective_task_id}] {text}")
                .trim()
                .to_string();
        }

        if text.is_empty() && is_completion {
            text = format!("[Task: {effective_task_id}]");
        }

        text
    }

    fn build_question_with_helper_barrier(
        &self,
        sender: &str,
        task_context: &TaskContext,
    ) -> Option<String> {
        if !task_context.has_existing_task_id {
            return Some(task_context.to_prompt());
        }

        let mut pending_helper_tasks = match self.pending_helper_tasks.lock() {
            Ok(pending_helper_tasks) => pending_helper_tasks,
            Err(poisoned) => poisoned.into_inner(),
        };

        let Some(pending_task) = pending_helper_tasks.get_mut(&task_context.task_id) else {
            return Some(task_context.to_prompt());
        };

        if !pending_task.expected_helpers.contains(sender) {
            return Some(task_context.to_prompt());
        }

        let response_body = task_context.original_message.trim().to_string();
        if response_body.is_empty() {
            println!(
                "helper response for {} from {} is empty, waiting for remaining helpers",
                task_context.task_id, sender
            );
            return None;
        }

        pending_task
            .responses
            .insert(sender.to_string(), response_body);

        let expected = pending_task.expected_helpers.len();
        let received = pending_task.responses.len();
        if received < expected {
            println!(
                "helper response progress for {}: {received}/{expected}, waiting",
                task_context.task_id
            );
            return None;
        }

        let mut aggregated_lines = pending_task
            .responses
            .iter()
            .map(|(helper, response)| format!("- {helper}: {response}"))
            .collect::<Vec<_>>();
        aggregated_lines.sort();

        let requester = pending_task.requester.clone();
        pending_helper_tasks.remove(&task_context.task_id);

        println!(
            "all helper responses received for {} (requester {}), continuing",
            task_context.task_id, requester
        );

        let combined_message = format!(
            "[CollectedHelperResponses]\n{}",
            aggregated_lines.join("\n")
        );
        Some(task_context.to_prompt_with_message(&combined_message))
    }

    fn update_pending_helper_task_state(
        &self,
        task_id: &str,
        requester: &str,
        response: &AgentResponse,
        mention_targets: &[MentionTarget],
    ) {
        let expected_helpers = mention_targets
            .iter()
            .filter(|target| target.source_tag.is_some())
            .map(|target| target.user_id.to_string())
            .collect::<HashSet<_>>();

        let mut pending_helper_tasks = match self.pending_helper_tasks.lock() {
            Ok(pending_helper_tasks) => pending_helper_tasks,
            Err(poisoned) => poisoned.into_inner(),
        };

        if !expected_helpers.is_empty() {
            let expected_count = expected_helpers.len();
            pending_helper_tasks.insert(
                task_id.to_string(),
                PendingHelperTask {
                    requester: requester.to_string(),
                    expected_helpers,
                    responses: HashMap::new(),
                },
            );
            println!(
                "tracking helper responses for task {task_id}: expecting {expected_count} helper(s)"
            );
            return;
        }

        if response.is_completion() && pending_helper_tasks.remove(task_id).is_some() {
            println!("cleared pending helper state for completed task {task_id}");
        }
    }

    async fn resolve_mentions(
        &self,
        response: &AgentResponse,
        task_context: &TaskContext,
        room: &Room,
    ) -> Vec<MentionTarget> {
        let mut mentions = self.resolve_user_mentions(&response.helper_names);
        mentions.extend(
            self.resolve_user_mentions_from_room_labels(room, &response.helper_names, &mentions)
                .await,
        );

        if response.is_completion() {
            let requester = response
                .requester
                .as_deref()
                .unwrap_or(&task_context.requester);
            let requester_user_id = UserId::parse(requester)
                .or_else(|_| UserId::parse(task_context.requester.as_str()));
            if let Ok(user_id) = requester_user_id {
                mentions.push(MentionTarget {
                    label: user_id.to_string(),
                    user_id,
                    source_tag: None,
                });
            } else {
                eprintln!(
                    "Skipping invalid requester user id '{}' and fallback '{}'",
                    requester, task_context.requester
                );
            }
        }

        mentions.sort_by(|left, right| left.user_id.cmp(&right.user_id));
        mentions.dedup_by(|left, right| left.user_id == right.user_id);
        mentions = mentions
            .into_iter()
            .filter(|x| x.user_id != room.own_user_id())
            .collect();
        for mention in &mut mentions {
            if let Some(room_label) = Self::lookup_room_label(room, mention.user_id.as_ref()).await
            {
                mention.label = room_label;
            }
        }

        mentions
    }

    async fn resolve_user_mentions_from_room_labels(
        &self,
        room: &Room,
        helper_names: &[String],
        existing_mentions: &[MentionTarget],
    ) -> Vec<MentionTarget> {
        let unresolved_helper_names = helper_names
            .iter()
            .filter(|helper_name| {
                !existing_mentions
                    .iter()
                    .any(|mention| mention.source_tag.as_deref() == Some(helper_name.as_str()))
            })
            .collect::<Vec<_>>();
        if unresolved_helper_names.is_empty() {
            return Vec::new();
        }

        let members = match room.members(RoomMemberships::ACTIVE).await {
            Ok(members) => members,
            Err(err) => {
                eprintln!("failed to list room members for mention resolution: {err}");
                return Vec::new();
            }
        };

        let mut mentions = Vec::new();
        for helper_name in unresolved_helper_names {
            let helper_label = helper_name.trim();
            if helper_label.is_empty() {
                continue;
            }
            let helper_label_lower = helper_label.to_lowercase();

            for member in &members {
                let display_name = member
                    .display_name()
                    .map(str::trim)
                    .filter(|display_name| !display_name.is_empty())
                    .unwrap_or_else(|| member.user_id().as_str());
                let display_name_lower = display_name.to_lowercase();
                if display_name_lower != helper_label_lower
                    && !display_name_lower.contains(&helper_label_lower)
                {
                    continue;
                }

                mentions.push(MentionTarget {
                    user_id: member.user_id().to_owned(),
                    label: display_name.to_string(),
                    source_tag: Some(helper_name.clone()),
                });
            }
        }

        mentions.sort_by(|left, right| left.user_id.cmp(&right.user_id));
        mentions.dedup_by(|left, right| left.user_id == right.user_id);
        mentions
    }

    fn extract_marker_value(message: &str, prefix: &str, suffix: char) -> Option<String> {
        let start = message.find(prefix)?;
        let after_prefix = &message[start + prefix.len()..];
        let end = after_prefix.find(suffix)?;
        let value = after_prefix[..end].trim();
        if value.is_empty() {
            return None;
        }
        Some(value.to_string())
    }

    fn strip_marker(message: &str, prefix: &str) -> String {
        let mut stripped = String::with_capacity(message.len());
        let mut remaining = message;

        while let Some(start) = remaining.find(prefix) {
            stripped.push_str(&remaining[..start]);
            let after_prefix = &remaining[start + prefix.len()..];
            let Some(end) = after_prefix.find(']') else {
                stripped.push_str(&remaining[start..]);
                return stripped;
            };
            remaining = &after_prefix[end + 1..];
        }

        stripped.push_str(remaining);
        stripped
    }

    fn resolve_user_mentions(&self, helper_names: &[String]) -> Vec<MentionTarget> {
        let mut mentions = Vec::new();

        for helper_name in helper_names {
            let mut resolved_any = false;
            for user_id in self.adk_agent.find_helper_user_ids_by_name(helper_name) {
                match UserId::parse(user_id.clone()) {
                    Ok(owned) => {
                        mentions.push(MentionTarget {
                            label: helper_name.clone(),
                            user_id: owned,
                            source_tag: Some(helper_name.clone()),
                        });
                        resolved_any = true;
                    }
                    Err(_) => {
                        eprintln!(
                            "Skipping invalid helper user id '{user_id}' for '{helper_name}'"
                        );
                    }
                }
            }

            if !resolved_any {
                let direct_user_id = if helper_name.starts_with('@') {
                    helper_name.clone()
                } else {
                    format!("@{helper_name}")
                };

                if let Ok(owned) = UserId::parse(direct_user_id) {
                    mentions.push(MentionTarget {
                        label: helper_name.clone(),
                        user_id: owned,
                        source_tag: Some(helper_name.clone()),
                    });
                }
            }
        }

        mentions.sort_by(|left, right| left.user_id.cmp(&right.user_id));
        mentions.dedup_by(|left, right| left.user_id == right.user_id);
        mentions
    }

    fn attach_visible_mentions(
        message: String,
        mention_targets: &[MentionTarget],
    ) -> (String, Option<String>) {
        if mention_targets.is_empty() {
            return (message, None);
        }

        let mut plain = message.trim().to_string();
        let mut html = Self::text_to_html(&plain);
        let mut used_ids: HashSet<OwnedUserId> = HashSet::new();
        let mut handled_tags: HashSet<String> = HashSet::new();

        for target in mention_targets
            .iter()
            .filter(|target| target.source_tag.is_some())
        {
            let Some(source_tag) = target.source_tag.as_ref() else {
                continue;
            };
            if !handled_tags.insert(source_tag.clone()) {
                continue;
            }

            let grouped = mention_targets
                .iter()
                .filter(|candidate| candidate.source_tag.as_deref() == Some(source_tag.as_str()))
                .collect::<Vec<_>>();

            if grouped.is_empty() {
                continue;
            }

            let placeholder = format!("@[{}]", source_tag);
            let bare_placeholder = format!("@{}", source_tag);
            let plain_mention = grouped
                .iter()
                .map(|candidate| format!("@[{}]", candidate.label))
                .collect::<Vec<_>>()
                .join(" ");
            let html_mention = grouped
                .iter()
                .map(|candidate| {
                    format!(
                        "<a href=\"https://matrix.to/#/{}\">@[{}]</a>",
                        candidate.user_id,
                        Self::escape_html(&candidate.label)
                    )
                })
                .collect::<Vec<_>>()
                .join(" ");

            if plain.contains(&placeholder) {
                plain = plain.replace(&placeholder, &plain_mention);
                html = html.replace(&placeholder, &html_mention);
                for candidate in grouped {
                    used_ids.insert(candidate.user_id.clone());
                }
            } else {
                let (updated_plain, replaced_plain) =
                    Self::replace_mention_token(&plain, &bare_placeholder, &plain_mention);
                if replaced_plain {
                    let (updated_html, replaced_html) =
                        Self::replace_mention_token(&html, &bare_placeholder, &html_mention);
                    if replaced_html {
                        plain = updated_plain;
                        html = updated_html;
                        for candidate in grouped {
                            used_ids.insert(candidate.user_id.clone());
                        }
                    }
                }
            }
        }

        let trailing = mention_targets
            .iter()
            .filter(|target| !used_ids.contains(&target.user_id))
            .collect::<Vec<_>>();

        if !trailing.is_empty() {
            let trailing_plain = trailing
                .iter()
                .map(|target| format!("@[{}]", target.label))
                .collect::<Vec<_>>()
                .join(" ");

            let trailing_html = trailing
                .iter()
                .map(|target| {
                    format!(
                        "<a href=\"https://matrix.to/#/{}\">@[{}]</a>",
                        target.user_id,
                        Self::escape_html(&target.label)
                    )
                })
                .collect::<Vec<_>>()
                .join(" ");

            if plain.is_empty() {
                plain = trailing_plain;
                html = trailing_html;
            } else {
                plain = format!("{plain} {trailing_plain}");
                html = format!("{html} {trailing_html}");
            }
        }

        if plain.is_empty() {
            return (plain, None);
        }

        (plain, Some(html))
    }

    fn text_to_html(input: &str) -> String {
        Self::escape_html(input).replace('\n', "<br>")
    }

    fn replace_mention_token(input: &str, token: &str, replacement: &str) -> (String, bool) {
        let mut replaced = false;
        let mut result = String::with_capacity(input.len());
        let mut remaining = input;

        while let Some(position) = remaining.find(token) {
            let prefix = &remaining[..position];
            let suffix = &remaining[position + token.len()..];

            if Self::is_mention_boundary_before(prefix.chars().next_back())
                && Self::is_mention_boundary_after(suffix.chars().next())
            {
                result.push_str(prefix);
                result.push_str(replacement);
                remaining = suffix;
                replaced = true;
            } else {
                let mut chars = remaining.chars();
                if let Some(ch) = chars.next() {
                    result.push(ch);
                    remaining = chars.as_str();
                } else {
                    break;
                }
            }
        }

        result.push_str(remaining);
        (result, replaced)
    }

    fn is_mention_boundary_before(ch: Option<char>) -> bool {
        ch.is_none_or(|value| !(value.is_alphanumeric() || matches!(value, '_' | '.' | '+' | '-')))
    }

    fn is_mention_boundary_after(ch: Option<char>) -> bool {
        ch.is_none_or(|value| !(value.is_alphanumeric() || matches!(value, '_' | '-' | '.' | ':')))
    }

    fn escape_html(input: &str) -> String {
        let mut escaped = String::with_capacity(input.len());
        for ch in input.chars() {
            match ch {
                '&' => escaped.push_str("&amp;"),
                '<' => escaped.push_str("&lt;"),
                '>' => escaped.push_str("&gt;"),
                '"' => escaped.push_str("&quot;"),
                '\'' => escaped.push_str("&#39;"),
                _ => escaped.push(ch),
            }
        }
        escaped
    }

    async fn lookup_room_label(room: &Room, user_id: &UserId) -> Option<String> {
        let member = match room.get_member(user_id).await {
            Ok(member) => member,
            Err(_) => return None,
        }?;

        member
            .display_name()
            .map(str::trim)
            .filter(|label| !label.is_empty())
            .map(ToOwned::to_owned)
    }
}

struct MentionTarget {
    user_id: OwnedUserId,
    label: String,
    source_tag: Option<String>,
}

struct PendingHelperTask {
    requester: String,
    expected_helpers: HashSet<String>,
    responses: HashMap<String, String>,
}

struct TaskMetadata {
    task_id: Option<String>,
    requester: Option<String>,
    body: String,
    had_task_id: bool,
}

impl TaskMetadata {
    fn parse(message: &str) -> Self {
        let task_id = MatrixAdkAgent::extract_marker_value(message, "[Task:", ']');
        let requester = MatrixAdkAgent::extract_marker_value(message, "[Requester:", ']');
        let without_requester = MatrixAdkAgent::strip_marker(message, "[Requester:");
        let body = MatrixAdkAgent::strip_marker(&without_requester, "[Task:")
            .trim()
            .to_string();

        Self {
            had_task_id: task_id.is_some(),
            task_id,
            requester,
            body,
        }
    }
}

struct TaskContext {
    task_id: String,
    requester: String,
    sender: String,
    room_id: String,
    original_message: String,
    has_existing_task_id: bool,
}

impl TaskContext {
    fn to_prompt_with_message(&self, original_message: &str) -> String {
        format!(
            "[TaskContext]\ntask_id: {}\nrequester: {}\nsender: {}\nroom_id: {}\noriginal_message:\n{}",
            self.task_id, self.requester, self.sender, self.room_id, original_message
        )
    }

    fn to_prompt(&self) -> String {
        self.to_prompt_with_message(&self.original_message)
    }
}

struct AgentResponse {
    text: String,
    helper_names: Vec<String>,
    task_id: Option<String>,
    completion_task_id: Option<String>,
    requester: Option<String>,
}

impl AgentResponse {
    fn is_completion(&self) -> bool {
        self.completion_task_id.is_some() || (self.helper_names.is_empty() && !self.text.is_empty())
    }

    fn effective_task_id(&self, task_context: &TaskContext) -> String {
        self.task_id
            .clone()
            .or(self.completion_task_id.clone())
            .unwrap_or_else(|| task_context.task_id.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::AgentResponse;
    use super::MatrixAdkAgent;
    use super::MentionTarget;
    use super::TaskContext;
    use matrix_sdk::ruma::OwnedUserId;
    use std::collections::HashMap;

    #[test]
    fn inviter_regex_supports_multiple_accounts() {
        let regex = regex::Regex::new(r"(@alice:matrix\.org|@bob:matrix\.org)").unwrap();

        assert!(super::inviter_matches(Some(&regex), "@alice:matrix.org"));
        assert!(super::inviter_matches(Some(&regex), "@bob:matrix.org"));
        assert!(!super::inviter_matches(Some(&regex), "@mallory:matrix.org"));
    }

    #[test]
    fn missing_inviter_regex_allows_all_accounts() {
        assert!(super::inviter_matches(None, "@anyone:matrix.org"));
    }

    #[test]
    fn extract_helper_tags_supports_bare_mentions() {
        let helper_names = MatrixAdkAgent::extract_helper_tags(
            "Please coordinate with @Agent-One and @[Agent Two].",
        );

        assert_eq!(helper_names, vec!["Agent Two", "Agent-One"]);
    }

    #[test]
    fn extract_helper_tags_ignores_email_addresses() {
        let helper_names = MatrixAdkAgent::extract_helper_tags(
            "Reach me at martin@example.com, then loop in @Agent-One.",
        );

        assert_eq!(helper_names, vec!["Agent-One"]);
    }

    #[test]
    fn attach_visible_mentions_upgrades_bare_helper_mentions() {
        let mention_targets = vec![MentionTarget {
            user_id: OwnedUserId::try_from("@agent-one:example.com").unwrap(),
            label: "Agent One".to_string(),
            source_tag: Some("Agent-One".to_string()),
        }];

        let (plain, html) = MatrixAdkAgent::attach_visible_mentions(
            "Please ask @Agent-One to review this.".to_string(),
            &mention_targets,
        );

        assert_eq!(plain, "Please ask @[Agent One] to review this.");
        assert_eq!(
            html.unwrap(),
            "Please ask <a href=\"https://matrix.to/#/@agent-one:example.com\">@[Agent One]</a> to review this."
        );
    }

    #[test]
    fn starts_with_task_failed_marker_matches_trimmed_prefix() {
        assert!(MatrixAdkAgent::starts_with_task_failed_marker(
            "  [TaskFailed: task-42] delegation loop detected"
        ));
    }

    #[test]
    fn starts_with_task_failed_marker_ignores_task_prefix() {
        assert!(MatrixAdkAgent::starts_with_task_failed_marker(
            "[Task: task-4] [TaskFailed: task-4] repeated"
        ));
    }

    #[test]
    fn starts_with_task_failed_marker_rejects_later_marker() {
        assert!(!MatrixAdkAgent::starts_with_task_failed_marker(
            "Please note [TaskFailed: task-42] delegation loop detected"
        ));
    }

    #[test]
    fn resolve_requester_for_task_map_keeps_original_requester() {
        let mut task_requesters = HashMap::new();

        let first = MatrixAdkAgent::resolve_requester_for_task_map(
            &mut task_requesters,
            "task-2",
            Some("@alice:matrix.org"),
            "@alice:matrix.org",
        );
        let second = MatrixAdkAgent::resolve_requester_for_task_map(
            &mut task_requesters,
            "task-2",
            None,
            "@charlie:matrix.org",
        );

        assert_eq!(first, "@alice:matrix.org");
        assert_eq!(second, "@alice:matrix.org");
    }

    #[test]
    fn resolve_requester_for_task_map_falls_back_to_sender_once() {
        let mut task_requesters = HashMap::new();

        let requester = MatrixAdkAgent::resolve_requester_for_task_map(
            &mut task_requesters,
            "task-7",
            None,
            "@bob:matrix.org",
        );

        assert_eq!(requester, "@bob:matrix.org");
        assert_eq!(
            task_requesters.get("task-7"),
            Some(&"@bob:matrix.org".to_string())
        );
    }

    #[test]
    fn task_context_to_prompt_with_message_replaces_body() {
        let context = TaskContext {
            task_id: "task-11".to_string(),
            requester: "@requester:example.com".to_string(),
            sender: "@helper:example.com".to_string(),
            room_id: "!room:example.com".to_string(),
            original_message: "old message".to_string(),
            has_existing_task_id: true,
        };

        let prompt = context.to_prompt_with_message("[CollectedHelperResponses]\n- @a: done");
        assert!(prompt.contains("task_id: task-11"));
        assert!(prompt.contains("requester: @requester:example.com"));
        assert!(prompt.contains("[CollectedHelperResponses]"));
        assert!(!prompt.contains("old message"));
    }

    #[test]
    fn agent_response_effective_task_id_prefers_explicit_task_marker() {
        let response = AgentResponse {
            text: "ok".to_string(),
            helper_names: Vec::new(),
            task_id: Some("task-3".to_string()),
            completion_task_id: Some("task-4".to_string()),
            requester: None,
        };
        let context = TaskContext {
            task_id: "task-1".to_string(),
            requester: "@requester:example.com".to_string(),
            sender: "@sender:example.com".to_string(),
            room_id: "!room:example.com".to_string(),
            original_message: "x".to_string(),
            has_existing_task_id: true,
        };

        assert_eq!(response.effective_task_id(&context), "task-3");
    }

    #[test]
    fn agent_response_effective_task_id_falls_back_to_context_task() {
        let response = AgentResponse {
            text: "ok".to_string(),
            helper_names: Vec::new(),
            task_id: None,
            completion_task_id: None,
            requester: None,
        };
        let context = TaskContext {
            task_id: "task-22".to_string(),
            requester: "@requester:example.com".to_string(),
            sender: "@sender:example.com".to_string(),
            room_id: "!room:example.com".to_string(),
            original_message: "x".to_string(),
            has_existing_task_id: false,
        };

        assert_eq!(response.effective_task_id(&context), "task-22");
    }
}
