use adk_rust::{
    AdkIdentity, Agent, AppName, Content, Part, SessionId, Tool, UserId,
    agent::LlmAgentBuilder,
    futures::{Stream, StreamExt as _},
    model::openai::{OpenAIResponsesClient, OpenAIResponsesConfig},
    runner::{Runner, RunnerConfigBuilder},
    serde_json::json,
    session::{
        AppendEventRequest, CreateRequest, Event, GetRequest, InMemorySessionService,
        SessionService,
    },
    tool::FunctionTool,
};

use anyhow::{Context, Result};
use std::{
    collections::HashMap,
    fs,
    path::PathBuf,
    sync::{Arc, Mutex},
};

const DEFAULT_INSTRUCTION: &str = "You are a helpful assistant.";
const APP_NAME: &str = "chatbot";
const ROOM_CONTEXT_SESSION_ID: &str = "room-context";
const COLLABORATION_PROTOCOL: &str = r#"When you receive a chat message, first inspect the known participant introductions
    with the get_known_introductions tool before deciding whether to answer yourself or collaborate. Use those introductions
    to judge who is best suited for the task.

    If you are asked to introduce yourself, use the introduce_yourself tool and return only its result, nothing more. Do not mention anyone then!

    You may receive a task context block containing task_id, requester, sender, room_id, and the original message. Keep the task_id stable across
    all collaboration messages.

    If another participant is better suited or collaboration is needed, respond with a message that includes `[Task: <task_id>]` and mention collaborators
    as `@[Agent-Name]`.

    If you complete the task yourself, include `[TaskComplete: <task_id>]` in your reply. When a requester is provided in the context, also include `[Requester: <requester_user_id>]`.

    When your own instructions say you must ask or consult another participant, always delegate first: do not answer on your own and do not include
    `[TaskComplete: <task_id>]` until you have received that participant's answer. A delegation message must be written exactly once and contain
    `[Task: <task_id>]`, one `@[Agent-Name]` mention and the concrete question for that participant, phrased so it is understandable without any other context.
    Use the `name` value of the participant's entry in the known introductions (exactly as listed, e.g. their Matrix user id) as the mention. Never mention the requester or the sender when delegating.

    Do not invent collaborators. Base delegation decisions on known introductions.

    Never, and I mean never, use your own name in your responses or mention yourself. Always refer to yourself as 'I' or 'me'.

    If you ever get the feeling that you are stuck in a loop (e.g. when being asked nearly exactly the same question again or giving the exactly same response) or you are unable to complete the task, respond with `[TaskFailed: <task_id>]` and provide a brief explanation of the issue. And in this case, DO NOT mention anyone, because it would open the loop again.

    If you have the feeling that mentioning you in the request message was not on purpose (e.g. you were not meant), then you can ignore the request and respond with a brief explanation. And in this case, DO NOT mention anyone, because it would open a loop again.

    Please answer short and precise, do not use markdown in your answer.
    "#;

const DEFAULT_INTRODUCTION: &str = "Hello! I am an AI assistant. How can I help you today?";

#[derive(Clone)]
pub struct AdkOpenAiAgentConfig {
    introduction: Option<String>,
    instruction: Option<String>,
    introductions_store_file: PathBuf,
    introduction_file: PathBuf,
    instruction_file: PathBuf,
    open_responses_model: String,
    open_responses_base_url: String,
    openai_api_key: String,
    tools: Vec<Arc<dyn Tool>>,
    agent: Option<Arc<dyn Agent>>,
}

#[derive(Clone)]
pub struct AdkOpenAiAgent {
    config: AdkOpenAiAgentConfig,
    introductions: Arc<Mutex<HashMap<String, String>>>,
    llm_agent: Arc<dyn Agent>,
    session_service: Arc<dyn SessionService>,
}

impl AdkOpenAiAgentConfig {
    pub fn default_from_env() -> Result<Self, anyhow::Error> {
        let api_key = std::env::var("OPENAI_API_KEY")
            .context("no OpenAI API Key given by env parameter OPENAI_API_KEY")?;
        let base_url = std::env::var("OPEN_RESPONSES_BASE_URL")
            .unwrap_or_else(|_| "https://api.openai.com/v1".to_string());
        let model =
            std::env::var("OPEN_RESPONSES_MODEL").unwrap_or_else(|_| "gpt-4.1-nano".to_string());
        let introductions_store_file = std::env::var("INTRODUCTIONS_STORE_FILE")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from(".matrix-store/introductions.json"));
        let introduction_file = std::env::var("INTRODUCTION_FILE")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("INTRODUCTION.md"));
        let instruction_file = introduction_file.with_file_name("INSTRUCTION.md");
        let instruction = std::env::var("INSTRUCTION")
            .ok()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty());
        Ok(Self {
            introduction: None,
            instruction,
            introduction_file,
            instruction_file,
            introductions_store_file,
            open_responses_model: model,
            open_responses_base_url: base_url,
            openai_api_key: api_key,
            tools: Vec::new(),
            agent: None,
        })
    }
    pub fn with_instruction(mut self, instruction: String) -> Self {
        self.instruction = Some(instruction);
        self
    }
    pub fn with_introduction(mut self, introduction: String) -> Self {
        self.introduction = Some(introduction);
        self
    }
    pub fn with_tools(mut self, tools: Vec<Arc<dyn Tool>>) -> Self {
        self.tools = tools;
        self
    }
    pub fn with_agent(mut self, agent: Arc<dyn Agent>) -> Self {
        self.agent = Some(agent);
        self
    }
}

impl AdkOpenAiAgent {
    pub async fn new(config: AdkOpenAiAgentConfig) -> Result<Self, anyhow::Error> {
        ensure_introductions_dir_exists(&config)?;

        let openai_responses_client = initialize_openai_client(&config)?;

        let mut instruction = get_configured_instruction(&config);

        instruction.push_str("\n\n");
        instruction.push_str(COLLABORATION_PROTOCOL);

        let introduction = get_configured_introduction(&config);
        let introduction_tool = Arc::new(FunctionTool::new(
            "introduce_yourself",
            "Return the configured self-introduction string.",
            move |_ctx, _args| {
                let introduction = format!("[Introduction]: {}", introduction.clone());
                async move { Ok(json!(introduction)) }
            },
        ));

        let introductions = Self::load_introductions(&config.introductions_store_file)?;
        let moving_introductions = Arc::new(Mutex::new(introductions.clone()));
        let stored_introductions = Arc::new(Mutex::new(introductions));
        let known_introductions_tool = Arc::new(FunctionTool::new(
            "get_known_introductions",
            "Return all known introductions. Optional arg 'name' filters by helper name (supports case-insensitive partial match).",
            move |_ctx, args| {
                let introductions = Arc::clone(&moving_introductions);
                async move {
                    let requested_name = args
                        .get("name")
                        .and_then(|value| value.as_str())
                        .map(str::trim)
                        .filter(|name| !name.is_empty())
                        .map(ToOwned::to_owned);

                    let introductions = match introductions.lock() {
                        Ok(introductions) => introductions,
                        Err(_) => {
                            return Ok(json!({
                                "requested_name": requested_name,
                                "matches": [],
                                "all": [],
                                "error": "failed to read introductions registry"
                            }));
                        }
                    };

                    let mut all_entries: Vec<(String, String)> = introductions
                        .iter()
                        .map(|(name, intro)| (name.clone(), intro.clone()))
                        .collect();
                    all_entries.sort_by(|a, b| a.0.cmp(&b.0));

                    let all_json: Vec<_> = all_entries
                        .iter()
                        .map(|(name, introduction)| {
                            json!({"name": name, "introduction": introduction})
                        })
                        .collect();

                    if let Some(requested_name) = requested_name {
                        let requested_lower = requested_name.to_lowercase();
                        let matches_json: Vec<_> = all_entries
                            .iter()
                            .filter(|(name, _)| {
                                name.to_lowercase() == requested_lower
                                    || name.to_lowercase().contains(&requested_lower)
                            })
                            .map(|(name, introduction)| {
                                json!({"name": name, "introduction": introduction})
                            })
                            .collect();

                        return Ok(json!({
                            "requested_name": requested_name,
                            "matches": matches_json,
                            "all": all_json,
                        }));
                    }

                    Ok(json!({ "all": all_json }))
                }
            },
        ));

        let session_service = Arc::new(InMemorySessionService::new());

        let llm_agent = LlmAgentBuilder::new("adk-openai-agent")
            .description("Agent wrapper around the OpenAI Responses client")
            .model(openai_responses_client.clone())
            .instruction(instruction)
            .tool(introduction_tool)
            .tool(known_introductions_tool)
            .build()?;

        Ok(Self {
            config: config,
            introductions: stored_introductions.clone(),
            llm_agent: Arc::new(llm_agent),
            session_service: session_service,
        })
    }

    pub fn remember_introduction(&self, name: String, introduction: String) {
        if let Ok(mut introductions) = self.introductions.lock() {
            introductions.insert(name, introduction);
            if let Err(err) =
                Self::persist_introductions(&self.config.introductions_store_file, &introductions)
            {
                eprintln!(
                    "Failed to persist introductions to {}: {err}",
                    self.config.introductions_store_file.display()
                );
            }
        }
    }

    pub fn find_helper_user_ids_by_name(&self, name: &str) -> Vec<String> {
        let requested = name.trim().to_lowercase();
        if requested.is_empty() {
            return Vec::new();
        }

        let Ok(introductions) = self.introductions.lock() else {
            return Vec::new();
        };

        let mut matches: Vec<String> = introductions
            .keys()
            .filter(|known_name| {
                let known = known_name.to_lowercase();
                known == requested || known.contains(&requested)
            })
            .cloned()
            .collect();
        matches.sort();
        matches.dedup();
        matches
    }

    pub fn is_known_helper_user_id(&self, user_id: &str) -> bool {
        let Ok(introductions) = self.introductions.lock() else {
            return false;
        };

        introductions.contains_key(user_id)
    }

    pub async fn ask(
        self: &Arc<Self>,
        room_id: String,
        message: String,
    ) -> Result<String, anyhow::Error> {
        let user_id = UserId::new(room_id)?;
        let session_id = SessionId::new(ROOM_CONTEXT_SESSION_ID)?;

        // Runner::run expects an existing session; create it once per room if missing.
        self.load_or_create_session(&user_id, &session_id).await?;

        // Create a runner for this session
        let runner = Runner::new(
            RunnerConfigBuilder::new()
                .agent(self.llm_agent.clone())
                .app_name(APP_NAME)
                .session_service(self.session_service.clone())
                .build_config(),
        )?;

        // Create user content for the prompt
        let user_content = Content::new("user").with_text(message);

        // Ask AI model
        let stream = runner.run(user_id, session_id, user_content).await?;

        // Send prompt and stream the response
        println!(
            "📤 Sending prompt to {}...",
            self.config.open_responses_base_url
        );
        println!();

        let mut received_content = false;

        let response_text = fetch_response_text(stream, &mut received_content).await;

        if received_content {
            println!();
        }
        Ok(response_text)
    }

    pub async fn record_observed_message(
        self: &Arc<Self>,
        room_id: String,
        message: String,
    ) -> Result<(), anyhow::Error> {
        let user_id = UserId::new(room_id)?;
        let session_id = SessionId::new(ROOM_CONTEXT_SESSION_ID)?;
        self.load_or_create_session(&user_id, &session_id).await?;

        let identity = AdkIdentity::new(AppName::new(APP_NAME)?, user_id, session_id);

        let mut event = Event::new("matrix-observed-message");
        event.author = "user".to_string();
        event.set_content(Content::new("user").with_text(message));

        self.session_service
            .append_event_for_identity(AppendEventRequest { identity, event })
            .await?;
        Ok(())
    }

    async fn load_or_create_session(
        self: &Arc<Self>,
        user_id: &UserId,
        session_id: &SessionId,
    ) -> Result<(), anyhow::Error> {
        Ok(
            match self
                .session_service
                .get(GetRequest {
                    app_name: APP_NAME.to_string(),
                    user_id: user_id.to_string(),
                    session_id: session_id.to_string(),
                    num_recent_events: None,
                    after: None,
                })
                .await
            {
                Ok(_) => {}
                Err(err) => {
                    if err.to_string().contains("session.not_found") {
                        self.session_service
                            .create(CreateRequest {
                                app_name: APP_NAME.to_string(),
                                user_id: user_id.to_string(),
                                session_id: Some(session_id.to_string()),
                                state: HashMap::new(),
                            })
                            .await?;
                    } else {
                        return Err(err.into());
                    }
                }
            },
        )
    }

    fn load_introductions(path: &PathBuf) -> Result<HashMap<String, String>, anyhow::Error> {
        let serialized = match fs::read(path) {
            Ok(contents) => contents,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(HashMap::new()),
            Err(err) => {
                return Err(err).with_context(|| {
                    format!("failed to read introductions file {}", path.display())
                });
            }
        };

        serde_json::from_slice(&serialized)
            .with_context(|| format!("failed to parse introductions file {}", path.display()))
    }

    fn persist_introductions(
        path: &PathBuf,
        introductions: &HashMap<String, String>,
    ) -> Result<(), anyhow::Error> {
        let serialized = serde_json::to_vec_pretty(introductions)
            .context("failed to serialize known introductions")?;
        fs::write(path, serialized)
            .with_context(|| format!("failed to write introductions file {}", path.display()))
    }
}

pub fn get_configured_introduction(config: &AdkOpenAiAgentConfig) -> String {
    let mut resulting_introduction = "".to_string();
    if let Some(introduction) = &config.introduction {
        resulting_introduction.push_str(&introduction);
    } else {
        if config.introduction_file.exists() {
            let file_introduction = match fs::read_to_string(&config.introduction_file) {
                Ok(file_introduction) => file_introduction,
                Err(e) => {
                    println!("[Error] Error reading introduction file {}", e.to_string());
                    DEFAULT_INTRODUCTION.to_string()
                }
            };
            print!(
                "Loaded introduction from {}: {}",
                config.introduction_file.display(),
                file_introduction
            );
            resulting_introduction.push_str(&file_introduction);
        } else {
            return DEFAULT_INTRODUCTION.to_string();
        }
    }
    resulting_introduction
}

pub fn get_configured_instruction(config: &AdkOpenAiAgentConfig) -> String {
    if let Some(instruction) = &config.instruction {
        return instruction.clone();
    }

    if config.instruction_file.exists() {
        let file_instruction = match fs::read_to_string(&config.instruction_file) {
            Ok(file_instruction) => file_instruction,
            Err(e) => {
                println!("[Error] Error reading instruction file {}", e);
                return DEFAULT_INSTRUCTION.to_string();
            }
        };

        if !file_instruction.trim().is_empty() {
            print!(
                "Loaded instruction from {}: {}",
                config.instruction_file.display(),
                file_instruction
            );
            return file_instruction;
        }
    }

    DEFAULT_INSTRUCTION.to_string()
}

fn initialize_openai_client(
    config: &AdkOpenAiAgentConfig,
) -> Result<Arc<OpenAIResponsesClient>, anyhow::Error> {
    let openai_responses_api_config =
        OpenAIResponsesConfig::new(&config.openai_api_key, &config.open_responses_model)
            .with_open_responses_mode(true)
            .with_reasoning_effort(adk_rust::model::ReasoningEffort::Low)
            .with_reasoning_summary(adk_rust::model::openai::ReasoningSummary::Concise)
            .with_base_url(&config.open_responses_base_url);
    let openai_responses_client =
        Arc::new(OpenAIResponsesClient::new(openai_responses_api_config)?);
    Ok(openai_responses_client)
}

fn ensure_introductions_dir_exists(config: &AdkOpenAiAgentConfig) -> Result<(), anyhow::Error> {
    Ok(
        if let Some(parent) = config.introductions_store_file.parent() {
            fs::create_dir_all(parent).with_context(|| {
                format!("failed to create introductions dir {}", parent.display())
            })?;
        },
    )
}

async fn fetch_response_text(
    mut stream: std::pin::Pin<
        Box<
            dyn Stream<
                    Item = std::prelude::v1::Result<
                        adk_rust::prelude::Event,
                        adk_rust::prelude::AdkError,
                    >,
                > + Send,
        >,
    >,
    received_content: &mut bool,
) -> String {
    let mut response_text = String::new();
    while let Some(response) = stream.next().await {
        match response {
            Ok(event) => {
                // Streaming responses contain text deltas on partial events. Some
                // providers also emit the completed text on a terminal event; do
                // not append that snapshot after already collecting the deltas.
                let accept_text = event.llm_response.partial || response_text.is_empty();
                if let Some(content) = event.content()
                    && accept_text
                {
                    for part in &content.parts {
                        if let Part::Text { text } = part
                            && !text.is_empty()
                        {
                            print!("{text}");
                            response_text.push_str(text);
                            *received_content = true;
                        }
                    }
                }
            }
            Err(e) => {
                // Gracefully handle errors that may occur with third-party
                // providers due to missing OpenAI-specific fields.
                eprintln!("⚠️  Stream error (may be expected with some providers): {e}");
            }
        }
    }
    dedupe_repeated_text(response_text)
}

// Some providers emit the same text twice (delta snapshot + final snapshot); collapse exact repeats.
fn dedupe_repeated_text(text: String) -> String {
    let trimmed = text.trim();
    let len = trimmed.len();
    if len >= 2 && len % 2 == 0 && trimmed.is_char_boundary(len / 2) {
        let (first, second) = trimmed.split_at(len / 2);
        if first == second {
            return first.to_string();
        }
    }
    text
}
