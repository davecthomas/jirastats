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

    def summarize_issues(self, project_key: str, allowed_statuses=None) -> str:
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
            "Please summarize the following Jira issues:\n",
            *readable_texts  # Insert each human-readable block
        ]
        prompt_text = "\n\n".join(prompt_lines)

        # 4. Send the prompt to OpenAI for summarization
        completion = self.openai_client.sendPrompt(prompt_text)
        return completion


def test_summarizer():
    """
    Test function to verify the Summarizer class functionality.
    Attempts to summarize issues for the Jira project specified in .env.
    """
    project_key = os.getenv("JIRA_PROJECT")
    if not project_key:
        print("No project key found in environment variable JIRA_PROJECT.")
        return

    # Optional: specify the statuses you want to filter (or pass None for all)
    statuses_to_include = ["In Progress", "Completed"]

    summarizer = Summarizer()
    summary = summarizer.summarize_issues(
        project_key, allowed_statuses=statuses_to_include)
    print("\n--- Summary of Jira Issues ---\n")
    print(summary)


if __name__ == "__main__":
    test_summarizer()
