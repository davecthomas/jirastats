# jirastats

Grab stats and summaries from Jira

# Wut

```mermaid
sequenceDiagram
    participant User
    participant Summarizer
    participant MyJira
    participant Jira
    participant MyOpenAI
    participant OpenAI

    User->>Summarizer: summarize_issues(project_key, allowed_statuses)
    activate Summarizer

    note over Summarizer: 1. Summarizer calls MyJira<br/>to fetch issues.

    Summarizer->>MyJira: get_all_issues_for_project(project_key, allowed_statuses)
    activate MyJira

    note over MyJira: 2. MyJira queries Jira’s REST API.
    MyJira->>Jira: HTTP GET /search (with JQL)
    activate Jira

    Jira-->>MyJira: JSON of issues
    deactivate Jira

    note over MyJira: 3. MyJira returns the raw<br/>list of issue dictionaries.

    MyJira-->>Summarizer: List[Dict] (issues)
    deactivate MyJira

    note over Summarizer: 4. Summarizer builds<br/>human-readable text<br/>for each issue.
    Summarizer->>MyJira: build_human_readable_issue_text(issue)
    activate MyJira
    MyJira-->>Summarizer: Plain text summary
    deactivate MyJira

    note over Summarizer: (Repeat for each issue)

    note over Summarizer: 5. Summarizer combines<br/>issue texts into one prompt.

    Summarizer->>MyOpenAI: sendPrompt(prompt_text)
    activate MyOpenAI

    note over MyOpenAI: 6. MyOpenAI calls OpenAI<br/>chat completions endpoint.
    MyOpenAI->>OpenAI: Chat Completions API<br/>with prompt
    activate OpenAI

    OpenAI-->>MyOpenAI: Summarized text
    deactivate OpenAI

    note over MyOpenAI: 7. MyOpenAI returns<br/>the final summary.

    MyOpenAI-->>Summarizer: summary string
    deactivate MyOpenAI

    note over Summarizer: 8. Summarizer returns the<br/>summaries to the user.
    Summarizer-->>User: summarized text
    deactivate Summarizer
```

# Install

python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt

# Settings

```
.env file should have
JIRA_API_TOKEN=
JIRA_USER=you@you.com
JIRA_CO_URL=your_org_name
JIRA_PROJECT=
OPENAI_API_KEY=
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_USER=
OPENAI_COMPLETIONS_MODEL=o3-mini
```
