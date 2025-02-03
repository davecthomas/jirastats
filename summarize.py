# summarize.py

import os
from .jirastats import get_all_issues_for_project
from .openai import MyOpenAI


class Summarizer:
    """
    A class that uses Jira data to generate a high-level summary of all issues
    within a specified Jira project. It fetches the issues via the `get_all_issues_for_project`
    function, then sends the aggregated issue data to OpenAI for summarization.
    """

    def __init__(self):
        """
        Initializes the Summarizer class by creating an instance of MyOpenAI
        for handling chat completions.
        """
        self.openai_client = MyOpenAI()

    def summarize_issues(self, project_key: str) -> str:
        """
        Fetches all issues for a given Jira project using the existing Jira logic
        (get_all_issues_for_project), compiles the issue list into a structured text prompt,
        and then sends the prompt to OpenAI to get a summary.

        Args:
            project_key (str): The Jira project key (e.g., "TEST").

        Returns:
            str: A summary of the issues provided by the OpenAI model.
        """
        # 1. Fetch all issues for the project
        issues = get_all_issues_for_project(project_key)

        if not issues:
            return "No issues found to summarize."

        # 2. Build a structured text prompt from the issues
        #    You can customize how detailed you want to be here
        prompt_lines = [
            "Summarize the following list of Jira issues:\n",
            "Format: KEY, SUMMARY, STATUS, ASSIGNEE\n\n",
        ]
        for issue in issues:
            issue_key = issue.get("key", "NoKey")
            summary = issue.get("summary", "NoSummary")
            status = issue.get("status", "NoStatus")
            assignee = issue.get("assignee", "Unassigned")
            prompt_lines.append(
                f"- {issue_key}: {summary} | Status: {status} | Assignee: {assignee}")

        prompt_text = "\n".join(prompt_lines)

        # 3. Send the prompt to OpenAI for summarization
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

    summarizer = Summarizer()
    summary = summarizer.summarize_issues(project_key)
    print("\n--- Summary of Jira Issues ---")
    print(summary)


if __name__ == "__main__":
    test_summarizer()
