# summarize.py

import os
from jirastats import MyJira
from myopenai import MyOpenAI


class Summarizer:
    """
    A class that uses Jira data to generate a high-level summary of all issues
    within a specified Jira project. It fetches the issues via the MyJira class,
    converts them to a human-readable format, and then sends the content
    to OpenAI for summarization.
    """

    def __init__(self):
        """
        Initializes the Summarizer class by creating an instance of MyOpenAI
        for handling chat completions, and a MyJira instance for fetching & formatting Jira data.
        """
        self.openai_client = MyOpenAI()
        self.jira_client = MyJira()

    def summarize_issues(self, project_key: str, prompt="Please summarize these issues", allowed_statuses=None) -> str:
        """
        Fetches all issues for a given Jira project using MyJira, converts each issue
        to a more human-readable text, and then sends all that text to OpenAI to get a summary.

        Args:
            project_key (str): The Jira project key (e.g., "TEST").
            allowed_statuses (List[str], optional): Filter issues by one or more statuses.

        Returns:
            str: A summary of the issues provided by the OpenAI model.
        """
        # 1. Fetch issues from MyJira, optionally filtered by status.
        issues = self.jira_client.get_all_issues_for_project(
            project_key, allowed_statuses=allowed_statuses
        )
        if not issues:
            return f"No issues found to summarize for project: {project_key}"

        # 2. Convert each issue to a human-readable string using build_human_readable_issue_text.
        readable_texts = []
        for issue in issues:
            text_block = self.jira_client.build_human_readable_issue_text(
                issue)
            readable_texts.append(text_block)

        # 3. Build the final prompt for OpenAI
        prompt_lines = [
            prompt,
            *readable_texts  # Insert each human-readable block
        ]
        prompt_text = "\n\n".join(prompt_lines)

        # 4. Send the prompt to OpenAI for summarization
        completion = self.openai_client.sendPrompt(prompt_text)
        return completion


def test_summarizer(project_key: str = None, prompt: str = None):
    """
    Test function to verify the Summarizer class functionality.
    Attempts to summarize issues for the Jira project specified in .env.
    """

    # Optional: specify the statuses you want to filter (or pass None for all)
    statuses_to_include = ["In Progress", "Open"]

    summarizer = Summarizer()
    summary = summarizer.summarize_issues(
        project_key, prompt, allowed_statuses=statuses_to_include)
    print(
        f"\n--- Summary of {project_key} Issues in status {statuses_to_include}---\n")
    print(summary)


def is_hex_environment():
    """
    Checks if the current runtime is in a Hex environment.
    """
    return os.getenv('HEX_PROJECT_ID') is not None


if __name__ == "__main__":
    # If you want to move this to Hex, you can put this prompt and the project key in user input fields

    if not is_hex_environment():
        prompt: str = """Please summarize the following Jira issues. 
            Categorize major areas of work using a coherent priority scheme for ordering these categorized areas. 
            For each category summary, conclude the summary with a list of JIRA issue keys that are exemplars of this category. 
            Do not list more than 10 issues per category, for brevity. 
            Output format: \n
            [Category 1 Name]: [Summary of issues]. [Noteworthy activity on these issues]. [KEY-123, KEY-124, KEY-125]
            [Categoryt 2 Name]: (etc)\n
            :\n """
        project_key = os.getenv("JIRA_PROJECT", "FOO")
    else:
        # This should be the name of the Hex input field for the project key
        project_key = user_input_project_key
        # This should be the name of the Hex input field for the prompt
        prompt = user_input_prompt

    test_summarizer(project_key, prompt)
