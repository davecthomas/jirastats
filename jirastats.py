import os
import time
from typing import Dict, Union
from typing import Optional
from requests.auth import HTTPBasicAuth
import requests
import json
import base64
from datetime import datetime, timedelta, timezone
import os
from typing import List, Tuple
import pandas as pd
import numpy as np
from numpy import busday_count, mean

import requests
from dateutil.relativedelta import relativedelta

JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN")
JIRA_CO_URL = os.getenv("JIRA_CO_URL")
JIRA_USER = os.getenv("JIRA_USER")
JIRA_PROJECT = os.getenv("JIRA_PROJECT")
JIRA_MAX_PAGE_SIZE = 100

# Base URL for your Jira instance
JIRA_BASE_URL = f"https://{JIRA_CO_URL}.atlassian.net/rest/api/3"


def get_business_days(start_date: datetime, end_date: datetime) -> int:
    # Convert datetime to date if they are in datetime format
    start_date = start_date.date() if isinstance(
        start_date, datetime) else start_date
    end_date = end_date.date() if isinstance(end_date, datetime) else end_date

    # Count business days. Returns an int
    return busday_count(start_date, end_date)


# Returns a float, since often issues are resolved in < 1 day
def get_days(start: datetime, end: datetime) -> float:
    # Calculate the time difference
    time_difference = end - start

    # Convert the time difference to days (including fractional days)
    days = time_difference / timedelta(days=1)

    return round(days, 2)


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


def add_dict_to_dataframe(data_dict: Dict[str, int], df: pd.DataFrame, col_name_key: str, col_name_value: str) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")

    # Convert the dictionary to a DataFrame with specified column names
    new_data = pd.DataFrame(list(data_dict.items()), columns=[
                            col_name_key, col_name_value])

    # Concatenate the new data with the existing DataFrame
    new_data = pd.concat([df, new_data], ignore_index=True)

    return new_data


def calculate_avg_days_open(issues_data: List[Dict], df: pd.DataFrame) -> pd.DataFrame:
    user_open_days = {}

    for issue in issues_data:
        assignee = issue['fields']['assignee']['displayName']
        assigned_date = datetime.strptime(
            issue['fields']['created'], '%Y-%m-%dT%H:%M:%S.%f%z')
        completed_date = datetime.strptime(
            issue['fields']['resolutiondate'], '%Y-%m-%dT%H:%M:%S.%f%z')

        # Calculate days between assigned_date and completed_date
        open_days: int = get_days(assigned_date, completed_date)

        if assignee in user_open_days:
            user_open_days[assignee].append(open_days)
        else:
            user_open_days[assignee] = [open_days]

    # Calculate average open days per user
    for user, days in user_open_days.items():
        avg_days = np.mean(days)
        df.loc[df['name'] == user, 'avg_days_open'] = avg_days

    return df


# def get_assigned_users(jira_project, time_since) -> Dict:
#     jira_org_name = os.getenv("JIRA_ORG_NAME")
#     jira_user = os.getenv("JIRA_USER")
#     jira_api_token = os.getenv("JIRA_API_TOKEN")
#     jira_base_url = JIRA_BASE_URL
#     start_at: int = 0
#     max_results: int = JIRA_MAX_PAGE_SIZE
#     total_issues: int = None
#     user_issue_count: Dict[str, int, int] = {}

#     # Format the time_since to the appropriate format for JIRA API
#     time_since_str = time_since.strftime("%Y-%m-%d %H:%M")

#     # JQL (Jira Query Language) for searching issues
#     jql = f"project = {jira_project} AND assignee is not EMPTY AND updated >= '{time_since_str}' AND resolutiondate >= '{time_since.strftime('%Y-%m-%d %H:%M')}'"

#     # Prepare the request headers
#     headers = {
#         "Authorization": f"Basic {jira_api_token}",
#         "Content-Type": "application/json"
#     }

#     # API endpoint for searching issues
#     search_url = f"{jira_base_url}/search"
#     issue_counter: int = 0
#     fields = "assignee,resolutiondate,created"

#     while total_issues is None or start_at < total_issues:
#         # Make the GET request with pagination
#         params: Dict = {'jql': jql, 'startAt': start_at,
#                         'maxResults': max_results, 'fields': fields}
#         response = request_exponential_backoff(search_url, params)

#         # Check if the request was successful
#         if response.status_code == 200:
#             data = response.json()
#             issues = data.get('issues', [])

#             # Update user-issue counts
#             for issue in issues:
#                 issue_counter += 1
#                 print(f"{issue_counter} of {total_issues}",
#                       end='\r', flush=True)
#                 # Get assignee and increment their count of assigned issues
#                 assignee = issue['fields']['assignee']
#                 if assignee:
#                     # Increment issue count and store
#                     assignee_username: str = assignee['displayName']
#                     user_issue_count[assignee_username] = user_issue_count.get(
#                         assignee_username, 0) + 1

#             # Update the total number of issues and increment start_at
#             total_issues = data.get('total', 0)
#             start_at += len(issues)
#         else:
#             print(
#                 f'Failed to fetch data: {response.status_code} - {response.text}')
#             break

#     print(
#         f'Scanned through {issue_counter} of {total_issues} issues since {since_date_str}')

#     return user_issue_count


def get_completed_issues_by_user(jira_project, time_since) -> List[Dict[str, float]]:
    start_at: int = 0
    max_results: int = JIRA_MAX_PAGE_SIZE
    total_issues: int = None
    issue_counter: int = 0
    user_issue_data: Dict[str, Dict[str, Union[int, float]]] = {}
    dict_user_issue_data_init: Dict[str, int] = {
        'created_issues': 0, 'completed_issues': 0, 'comments': 0, 'total_days': 0}

    # JQL (Jira Query Language) for searching issues
    # Get all issues either completed or in progress that were updated or resolved in the lookback period
    jql = f"project = {jira_project} AND (status = 'Completed' OR status = 'In Progress') AND (updated >= '{time_since.strftime('%Y-%m-%d %H:%M')}' OR resolutiondate >= '{time_since.strftime('%Y-%m-%d %H:%M')}')"
    fields = "creator,assignee,resolutiondate,created,comment,status"

    # API endpoint for searching issues
    search_url = f"{JIRA_BASE_URL}/search"

    while total_issues is None or start_at < total_issues:
        params: Dict = {'jql': jql, 'startAt': start_at,
                        'maxResults': max_results, 'fields': fields}
        response = request_exponential_backoff(search_url, params)

        if response and response.status_code == 200:
            data = response.json()
            issues = data.get('issues', [])

            for issue in issues:
                issue_counter += 1
                print(f"{issue_counter} of {total_issues}",
                      end='\r', flush=True)
                creator = issue['fields'].get('creator')
                if creator:
                    creator_username: str = creator['displayName']
                    # Initialize a new dict entry if this is the first we've seen of the assignee
                    if creator_username not in user_issue_data:
                        user_issue_data[creator_username] = dict_user_issue_data_init
                    user_issue_data[creator_username]['created_issues'] += 1

                # The assignee only gets credit for the issue if it's completed
                assignee = issue['fields'].get('assignee')
                status = issue['fields'].get('status')
                if assignee and status and status["name"] == "Completed":
                    assignee_username: str = assignee['displayName']
                    created_date = datetime.strptime(
                        issue['fields']['created'], '%Y-%m-%dT%H:%M:%S.%f%z')
                    resolution_date = datetime.strptime(
                        issue['fields']['resolutiondate'], '%Y-%m-%dT%H:%M:%S.%f%z')
                    days_open = (resolution_date - created_date).days
                    # Initialize a new dict entry if this is the first we've seen of the assignee
                    if assignee_username not in user_issue_data:
                        user_issue_data[assignee_username] = dict_user_issue_data_init

                    user_issue_data[assignee_username]['total_days'] += days_open
                    user_issue_data[assignee_username]['completed_issues'] += 1

                # Get comments and add to dictionary
                comments = issue['fields'].get(
                    'comment', {}).get('comments', [])
                for comment in comments:
                    created = datetime.strptime(
                        comment['created'], '%Y-%m-%dT%H:%M:%S.%f%z').replace(tzinfo=None)
                    if created >= since_date:
                        comment_author: str = comment['author']['displayName']
                        # Initialize a new dict entry if this is the first we've seen of the commenter
                        if comment_author not in user_issue_data:
                            user_issue_data[comment_author] = dict_user_issue_data_init
                        user_issue_data[comment_author]['comments'] += 1

            total_issues = data.get('total', 0)
            start_at += len(issues)
        else:
            print(
                f'Failed to fetch data: {response.status_code} - {response.text}')
            break

    # Calculate average open days and prepare the result
    result = []
    for user, data in user_issue_data.items():
        avg_days = data['total_days'] / \
            data['completed_issues'] if data['completed_issues'] > 0 else 0
        result.append({
            'display_name': user,
            'avg_days': round(avg_days, 2),
            'created_issues': data['created_issues'],
            'completed_issues': data['completed_issues'],
            'comments': data['comments']
        })

    return result


if __name__ == "__main__":
    # Get the env settings
    months_lookback = int(os.getenv("DEFAULT_MONTHS_LOOKBACK", 3))
    if months_lookback < 1:
        months_lookback = 3

    default_lookback = int(os.getenv('DEFAULT_MONTHS_LOOKBACK', 3))
    since_date: datetime = get_date_months_ago(default_lookback)
    since_date_str: str = since_date.strftime('%Y-%m-%d')

    # dict_assigned_users: Dict = get_assigned_users(JIRA_PROJECT, since_date)
    # df_assigned_users = add_dict_to_dataframe(
    #     dict_assigned_users, pd.DataFrame(), "name", "assigned_issues")

    completed_issues_dict = get_completed_issues_by_user(
        JIRA_PROJECT, since_date)
    df_completed_issues = pd.DataFrame(completed_issues_dict)
    # Rename 'display_name' to 'name' to match the existing DataFrame
    df_completed_issues.rename(columns={'display_name': 'name'}, inplace=True)

    # # Merge the two DataFrames on the 'name' column
    # merged_df = pd.merge(
    #     df_assigned_users, df_completed_issues, on='name', how='outer')

    # Replace NaN values with 0 (in case there are users with assigned issues but no completed issues, and vice versa)
    df_completed_issues.fillna({'assigned_issues': 0, 'avg_days': 0, 'completed_issues': 0,
                                'comments': 0}, inplace=True)

    # I think this forces these columns to int from float
    df_completed_issues['created_issues'] = df_completed_issues['created_issues'].astype(
        int)
    df_completed_issues['completed_issues'] = df_completed_issues['completed_issues'].astype(
        int)

    # Calculate business days since since_date
    business_days = get_business_days(since_date, datetime.now())

    # Calculate issues closed per day
    df_completed_issues['created_issues_per_day'] = (
        df_completed_issues['created_issues'] / business_days).round(2)
    df_completed_issues['completed_issues_per_day'] = (
        df_completed_issues['completed_issues'] / business_days).round(2)
    df_completed_issues['comments_per_day'] = (
        df_completed_issues['comments'] / business_days).round(2)
    # print(df_completed_issues)

    # Format datetime as a string without seconds or timezone
    formatted_datetime = datetime.now().strftime('%Y-%m-%d_%H-%M')
    csv_file_path = f'{formatted_datetime}_{JIRA_PROJECT}_Since_{since_date_str}_Stats.csv'
    df_completed_issues.to_csv(csv_file_path, index=False)


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
