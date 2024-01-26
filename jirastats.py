import os
import time
from typing import Dict
from requests.auth import HTTPBasicAuth
import requests
import json
import base64
from datetime import datetime, timezone
import os
from typing import List, Tuple
import pandas as pd
import numpy as np

import requests
from dateutil.relativedelta import relativedelta

JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN")
JIRA_CO_URL = os.getenv("JIRA_CO_URL")
JIRA_USER = os.getenv("JIRA_USER")
JIRA_PROJECT = os.getenv("JIRA_PROJECT")
JIRA_MAX_PAGE_SIZE = 100

# Base URL for your Jira instance
JIRA_BASE_URL = f"https://{JIRA_CO_URL}.atlassian.net/rest/api/3"


def get_date_months_ago(months_ago) -> datetime:
    current_date = datetime.now()
    date_months_ago = current_date - relativedelta(months=months_ago)
    return date_months_ago


def encode_key(user_email, api_key):
    string_to_encode = f"{user_email}:{api_key}"
    encoded_bytes = base64.b64encode(string_to_encode.encode("utf-8"))
    encoded_string = str(encoded_bytes, "utf-8")
    return encoded_string


def request_exponential_backoff(url: str, params: Dict = None):
    exponential_backoff_retry_delays_list: list[int] = [1, 2, 4, 8, 16]
    headers = {
        "Authorization": f"Basic {JIRA_API_TOKEN}",
        "Content-Type": "application/json"
    }

    response = requests.get(url, auth=HTTPBasicAuth(
        JIRA_USER, JIRA_API_TOKEN), headers=headers, params=params)
    if response.status_code != 200:
        if response.status_code == 429 or response.status_code == 202:  # Try again
            for retry_attempt_delay in exponential_backoff_retry_delays_list:
                retry_url = response.headers.get('Location')
                # Wait for n seconds before checking the status
                time.sleep(retry_attempt_delay)
                retry_response_url: str = retry_url if retry_url else url
                print(
                    f"Retrying request for {retry_response_url} after {retry_attempt_delay} sec due to {response.status_code} response")
                response = requests.get(
                    retry_response_url, headers=headers)

                # Check if the retry response is 200
                if response.status_code == 200:
                    break  # Exit the loop on successful response

    if response.status_code == 200:
        return response
    else:
        return None


def get_assigned_users(jira_project, time_since):
    jira_org_name = os.getenv("JIRA_ORG_NAME")
    jira_user = os.getenv("JIRA_USER")
    jira_api_token = os.getenv("JIRA_API_TOKEN")
    jira_base_url = JIRA_BASE_URL
    start_at: int = 0
    max_results: int = JIRA_MAX_PAGE_SIZE
    total_issues: int = None
    user_issue_count: Dict[str, int] = {}

    # Format the time_since to the appropriate format for JIRA API
    time_since_str = time_since.strftime("%Y-%m-%d %H:%M")

    # JQL (Jira Query Language) for searching issues
    jql = f"project = {jira_project} AND assignee is not EMPTY AND updated >= '{time_since_str}'"

    # Prepare the request headers
    headers = {
        "Authorization": f"Basic {jira_api_token}",
        "Content-Type": "application/json"
    }

    # API endpoint for searching issues
    search_url = f"{jira_base_url}/search"

    while total_issues is None or start_at < total_issues:
        # Make the GET request with pagination
        params: Dict = {'jql': jql, 'startAt': start_at,
                        'maxResults': max_results}
        response = request_exponential_backoff(search_url, params)
        # Make the GET request with pagination
        # response = requests.get(search_url, auth=HTTPBasicAuth(jira_user, jira_api_token), headers=headers,
        #                         params={'jql': jql, 'startAt': start_at, 'maxResults': max_results})

        # Check if the request was successful
        if response.status_code == 200:
            data = response.json()
            issues = data.get('issues', [])

            # Update user-issue counts
            for issue in issues:
                assignee = issue['fields']['assignee']
                if assignee:
                    # or 'emailAddress' depending on your JIRA setup
                    assignee_username: str = assignee['displayName']
                    user_issue_count[assignee_username] = user_issue_count.get(
                        assignee_username, 0) + 1

            # Update the total number of issues and increment start_at
            total_issues = data.get('total', 0)
            start_at += len(issues)
        else:
            print(
                f'Failed to fetch data: {response.status_code} - {response.text}')
            break

    # # Make the request to Jira API
    # response = requests.get(search_url, auth=HTTPBasicAuth(
    #     jira_user, jira_api_token), headers=headers, params={"jql": jql})
    # # Raises an HTTPError if the HTTP request returned an unsuccessful status code
    # response.raise_for_status()

    # assigned_users: [] = None
    # # Parse response
    # issues_response = response.json()
    # if "issues" in issues_response:
    #     issues_list = issues_response["issues"]
    #     if isinstance(issues_list, list):
    #         assigned_users = {issue["fields"]["assignee"]["displayName"]
    #                           for issue in issues_list if "assignee" in issue["fields"]}
        print(f'Scanned through {total_issues} issues since {since_date_str}')

    return user_issue_count


if __name__ == "__main__":
    # Get the env settings
    months_lookback = int(os.getenv("DEFAULT_MONTHS_LOOKBACK", 3))
    if months_lookback < 1:
        months_lookback = 3

    default_lookback = int(os.getenv('DEFAULT_MONTHS_LOOKBACK', 3))
    since_date: datetime = get_date_months_ago(default_lookback)
    since_date_str: str = since_date.strftime('%Y-%m-%d')

    list_assigned_users: Dict = get_assigned_users(JIRA_PROJECT, since_date)
    print(f'Assigned users: {list_assigned_users}')


# # Endpoint to get issues for a sprint
# SPRINT_ISSUES_ENDPOINT = "/search"


# # JQL query to get issues that were added to the sprint at the beginning
# JQL_ISSUES_AT_START_OF_SPRINT = "project = \"Consumer Apps\" and \"Team[Dropdown]\" = Frontend and (\"Flow[Dropdown]\" = Onboarding or \"Flow[Dropdown]\" = Growth) and Sprint in (openSprints()) and (status changed to  (\"To Do\", Open, \"In Progress\", Reopened) during ('2023-04-03 00:00', '2023-04-04 23:59') or assignee changed during ('2023-04-03 00:00', '2023-04-04 23:59'))"

# # JQL query to get issues that were added to the sprint at the end
# JQL_ISSUES_AT_END_OF_SPRINT = "project = \"Consumer Apps\" and \"Team[Dropdown]\" = Frontend and (\"Flow[Dropdown]\" = Onboarding or \"Flow[Dropdown]\" = Growth) and Sprint in (openSprints())"

# # Set up headers with the encoded credentials
# headers = {
#     "Authorization": f"Basic {encode_key(JIRA_USER, JIRA_API_TOKEN)}",
#     "Content-Type": "application/json"
# }

# # Get the issues at the start of the sprint
# start_response = requests.get(
#     f"{JIRA_BASE_URL}{SPRINT_ISSUES_ENDPOINT}",
#     headers=headers,
#     params={"jql": JQL_ISSUES_AT_START_OF_SPRINT}
# )
# start_issues = json.loads(start_response.text)

# # Get the issues at the end of the sprint
# end_response = requests.get(
#     f"{JIRA_BASE_URL}{SPRINT_ISSUES_ENDPOINT}",
#     headers=headers,
#     params={"jql": JQL_ISSUES_AT_END_OF_SPRINT}
# )
# end_issues = json.loads(end_response.text)

# # Calculate turbulence
# turbulence = len(end_issues["issues"]) / len(start_issues["issues"])

# print(f"Turbulence: {turbulence}")
