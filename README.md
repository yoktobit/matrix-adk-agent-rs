# Matrix ADK Agent

A Matrix bot that uses the OpenAI Responses API, or an OpenAI-compatible Responses endpoint, to answer messages in rooms where it is mentioned. The bot automatically joins rooms when invited, persists its Matrix login session, and keeps a small registry of known agent introductions.

## Requirements

- Rust and Cargo
- A Matrix account for the bot
- An OpenAI API key, or credentials accepted by the configured OpenAI-compatible endpoint
- Access to a Matrix homeserver

## Configuration

The application loads a `.env` file from the current working directory before reading configuration. Environment variables can also be supplied directly by the shell or deployment system.

### Required variables

| Variable | Description |
| --- | --- |
| `MATRIX_USERNAME` | Username used to log in to Matrix. |
| `MATRIX_PASSWORD` | Password used to log in to Matrix. Keep this secret. |
| `OPENAI_API_KEY` | API key passed to the Responses API client. This is required even when using a custom OpenAI-compatible endpoint. |

### Optional variables

| Variable | Default | Description |
| --- | --- | --- |
| `MATRIX_HOMESERVER_URL` | `https://matrix.org` | Matrix homeserver URL. |
| `MATRIX_STORE_DIR` | `.matrix-store` | Directory used by the Matrix SDK SQLite store. Created automatically. |
| `MATRIX_SESSION_FILE` | `$MATRIX_STORE_DIR/session.json` | File used to save and restore the Matrix login session. If unset, the path is `.matrix-store/session.json`. |
| `MATRIX_DEVICE_DISPLAY_NAME` | `autojoin bot` | Display name registered for a new Matrix device. |
| `MATRIX_AUTOJOIN_INVITER_REGEX` | unset (accept all inviters) | Regular expression matched against the full Matrix user ID of the account inviting the bot. Invitations from non-matching accounts are ignored. Use alternation such as `(@alice:matrix.org|@bob:matrix.org)` for multiple accounts. |
| `OPEN_RESPONSES_BASE_URL` | `https://api.openai.com/v1` | Base URL for the OpenAI Responses API or a compatible provider. |
| `OPEN_RESPONSES_MODEL` | `gpt-4.1-nano` | Model name sent to the configured Responses endpoint. |
| `INSTRUCTION` | Contents of `INSTRUCTION.md`, or `You are a helpful assistant.` | System instruction for the LLM. A non-empty environment value takes precedence over the instruction file. The built-in collaboration protocol is appended automatically. |
| `INTRODUCTION_FILE` | `INTRODUCTION.md` | File containing the bot's self-introduction. If the file does not exist, a built-in introduction is used. The instruction file is derived from this path by replacing its filename with `INSTRUCTION.md`. |
| `INTRODUCTIONS_STORE_FILE` | `.matrix-store/introductions.json` | JSON file used to persist introductions observed from other agents. Its parent directory is created automatically. |

`MATRIX_SESSION_FILE` is independent of `MATRIX_STORE_DIR` when explicitly set. Relative paths are resolved from the process's current working directory.

## Example `.env`

```dotenv
MATRIX_HOMESERVER_URL=https://matrix.example.org
MATRIX_USERNAME=ai-bot
MATRIX_PASSWORD=replace-with-a-secret
MATRIX_DEVICE_DISPLAY_NAME=Matrix ADK Bot
# Optional: only autojoin invitations from these accounts
# MATRIX_AUTOJOIN_INVITER_REGEX=(@alice:matrix.org|@bob:matrix.org)

OPENAI_API_KEY=replace-with-an-api-key
OPEN_RESPONSES_BASE_URL=https://api.openai.com/v1
OPEN_RESPONSES_MODEL=gpt-4.1-nano

# Optional prompt and persistence overrides
# INSTRUCTION=You are a concise and helpful Matrix assistant.
# INTRODUCTION_FILE=./config/INTRODUCTION.md
# INTRODUCTIONS_STORE_FILE=./.matrix-store/introductions.json
# MATRIX_STORE_DIR=./.matrix-store
# MATRIX_SESSION_FILE=./.matrix-store/session.json
```

Do not commit `.env`, Matrix passwords, API keys, or saved session files. The repository's `.gitignore` already excludes `.env` and `.matrix-store`.

## Running

From this directory:

```bash
cargo run
```

On the first run, the bot logs in with `MATRIX_USERNAME` and `MATRIX_PASSWORD` and saves the Matrix session to `MATRIX_SESSION_FILE`. Later runs restore that session when possible, so the password is not sent on every startup.

The bot must be invited to a room. By default, it autojoins invitations from any account. Set `MATRIX_AUTOJOIN_INVITER_REGEX` to restrict this, for example `(@alice:matrix.org|@bob:matrix.org)`. The regex is matched against the inviter's complete Matrix user ID. After joining, the bot records observed text messages and asks the LLM for a response when it is mentioned.

## Prompt files

- `INTRODUCTION.md` controls the text returned when the bot is asked to introduce itself.
- `INSTRUCTION.md` supplies the base LLM instruction when `INSTRUCTION` is not set.
- The collaboration protocol used by the agent is always appended to the configured instruction.

To use a different pair of files, set `INTRODUCTION_FILE`; the corresponding instruction file is expected beside it with the filename `INSTRUCTION.md`. Set `INSTRUCTION` directly if only the instruction needs to be overridden.

## Troubleshooting

- **Missing configuration:** startup reports the name of each missing required variable.
- **Login happens on every run:** check that `MATRIX_SESSION_FILE` is writable and that the saved session file is preserved between deployments.
- **The bot cannot connect:** verify `MATRIX_HOMESERVER_URL`, account credentials, and network access.
- **The model endpoint fails:** verify `OPENAI_API_KEY`, `OPEN_RESPONSES_BASE_URL`, and `OPEN_RESPONSES_MODEL` for the selected provider.
