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
from scipy.stats import norm
from requests.models import Response

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

# Sleep seconds from now to the future time passed in


def sleep_until_ratelimit_reset_time(reset_epoch_time):
    # Convert the reset time from Unix epoch time to a datetime object
    reset_time = datetime.utcfromtimestamp(reset_epoch_time)

    # Get the current time
    now = datetime.utcnow()

    # Calculate the time difference
    time_diff = reset_time - now

    # Check if the sleep time is negative, which can happen if the reset time has passed
    if time_diff.total_seconds() < 0:
        print("\tNo sleep required. The rate limit reset time has already passed.")
    else:
        time_diff = timedelta(seconds=int(time_diff.total_seconds()))
        # Print the sleep time using timedelta's string representation
        print(f"\tSleeping until rate limit reset: {time_diff}")
        time.sleep(time_diff.total_seconds())
    return

# Check if we overran our rate limit. Take a short nap if so.
# Return True if we overran


def check_API_rate_limit(response: Response) -> bool:
    if response.status_code == 403 and 'X-Ratelimit-Remaining' in response.headers:
        if int(response.headers['X-Ratelimit-Remaining']) == 0:
            print(
                f"\t403 forbidden response header shows X-Ratelimit-Remaining at {response.headers['X-Ratelimit-Remaining']} requests.")
            sleep_until_ratelimit_reset_time(
                int(response.headers['X-RateLimit-Reset']))
    return (response.status_code == 403 and 'X-Ratelimit-Remaining' in response.headers)


# Retry backoff in 422, 202, or 403 (rate limit exceeded) responses
def jira_request_exponential_backoff(url: str, params: Dict = None):
    exponential_backoff_retry_delays_list: list[int] = [1, 2, 4, 8, 16, 32, 64]
    headers = {
        "Authorization": f"Basic {JIRA_API_TOKEN}",
        "Content-Type": "application/json"
    }

    retry: bool = False
    response: Response = Response()
    retry_url: str = None

    try:
        response = requests.get(url, auth=HTTPBasicAuth(
            JIRA_USER, JIRA_API_TOKEN), headers=headers, params=params)
    except requests.exceptions.Timeout:
        print("Initial request timed out.")
        retry = True

    if retry or (response is not None and response.status_code != 200):
        if response.status_code == 422 and response.reason == "Unprocessable Entity":
            dict_error: Dict[str, any] = json.loads(response.text)
            print(
                f"Skipping: {response.status_code} {response.reason} for url {url}\n\t{dict_error['message']}\n\t{dict_error['errors'][0]['message']}")

        elif retry or response.status_code == 202 or response.status_code == 403:  # Try again
            for retry_attempt_delay in exponential_backoff_retry_delays_list:
                if 'Location' in response.headers:
                    retry_url = response.headers.get('Location')
                # The only time we override the exponential backoff if we are asked by Github to wait
                if 'Retry-After' in response.headers:
                    retry_attempt_delay = response.headers.get('Retry-After')
                # Wait for n seconds before checking the status
                time.sleep(retry_attempt_delay)
                retry_response_url: str = retry_url if retry_url else url
                print(
                    f"Retrying request for {retry_response_url} after {retry_attempt_delay} sec due to {response.status_code} response")
                # A 403 may require us to take a nap
                check_API_rate_limit(response)

                try:
                    response = requests.get(
                        retry_response_url, headers=headers)
                except requests.exceptions.Timeout:
                    print(
                        f"Retry request timed out. retrying in {retry_attempt_delay} seconds.")
                    continue
                # Check if the retry response is 200
                if response.status_code == 200:
                    break  # Exit the loop on successful response
                else:
                    print(
                        f"\tRetried request and still got bad response status code: {response.status_code}")

    if response.status_code == 200:
        # print(f"Retry successful. Status code: {response.status_code}")
        return response.json()
    else:
        check_API_rate_limit(response)
        print(
            f"Retries exhausted. Giving up. Status code: {response.status_code}")
        return None


# def request_exponential_backoff(url: str, params: Dict = None):
#     exponential_backoff_retry_delays_list: list[int] = [1, 2, 4, 8, 16]
#     headers = {
#         "Authorization": f"Basic {JIRA_API_TOKEN}",
#         "Content-Type": "application/json"
#     }

#     response = requests.get(url, auth=HTTPBasicAuth(
#         JIRA_USER, JIRA_API_TOKEN), headers=headers, params=params)
#     if response.status_code != 200:
#         if response.status_code == 429 or response.status_code == 202:  # Try again
#             for retry_attempt_delay in exponential_backoff_retry_delays_list:
#                 retry_url = response.headers.get('Location')
#                 # Wait for n seconds before checking the status
#                 time.sleep(retry_attempt_delay)
#                 retry_response_url: str = retry_url if retry_url else url
#                 print(
#                     f"Retrying request for {retry_response_url} after {retry_attempt_delay} sec due to {response.status_code} response")
#                 response = requests.get(
#                     retry_response_url, headers=headers)

#                 # Check if the retry response is 200
#                 if response.status_code == 200:
#                     break  # Exit the loop on successful response

#     if response.status_code == 200:
#         return response
#     else:
#         return None


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


def get_completed_issues_by_user(jira_project, time_since) -> List[Dict[str, float]]:
    start_at: int = 0
    max_results: int = JIRA_MAX_PAGE_SIZE
    total_issues: int = None
    issue_counter: int = 0
    user_issue_data: Dict[str, Dict[str, Union[int, float]]] = {}
    dict_user_issue_data_init: Dict[str, int] = {
        'created_issues': 0, 'completed_issues': 0, 'comments': 0, 'total_days': 0, 'first_seen': time_since}

    # JQL (Jira Query Language) for searching issues
    # Get all issues either completed or in progress that were updated or resolved in the lookback period
    jql: str = f"project = {jira_project} AND (status = 'Completed' OR status = 'In Progress') AND (updated >= '{time_since.strftime('%Y-%m-%d %H:%M')}' OR resolutiondate >= '{time_since.strftime('%Y-%m-%d %H:%M')}')"
    fields: str = "creator,assignee,resolutiondate,created,comment,status"
    expand: str = "changelog"

    # API endpoint for searching issues
    search_url: str = f"{JIRA_BASE_URL}/search"

    while total_issues is None or start_at < total_issues:
        params: Dict = {'jql': jql, 'startAt': start_at,
                        'maxResults': max_results, 'fields': fields, 'expand': expand}
        response = jira_request_exponential_backoff(search_url, params)

        if response:
            data = response
            issues = data.get('issues', [])

            for issue in issues:
                issue_counter += 1
                print(f"{issue_counter} of {total_issues}",
                      end='\r', flush=True)

                # Grab creator stats
                creator = issue['fields'].get('creator')
                if creator:
                    creator_username: str = creator['displayName']
                    # Initialize a new dict entry if this is the first we've seen of the assignee
                    if creator_username not in user_issue_data:
                        user_issue_data[creator_username] = dict_user_issue_data_init.copy(
                        )
                    user_issue_data[creator_username]['created_issues'] += 1

                # Grab assignee stats
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
                        user_issue_data[assignee_username] = dict_user_issue_data_init.copy(
                        )

                    user_issue_data[assignee_username]['total_days'] += days_open
                    user_issue_data[assignee_username]['completed_issues'] += 1

                # Get commenter stats
                comments = issue['fields'].get(
                    'comment', {}).get('comments', [])
                for comment in comments:
                    created = datetime.strptime(
                        comment['created'], '%Y-%m-%dT%H:%M:%S.%f%z').replace(tzinfo=None)
                    if created >= since_date:
                        comment_author: str = comment['author']['displayName']
                        # Initialize a new dict entry if this is the first we've seen of the commenter
                        if comment_author not in user_issue_data:
                            user_issue_data[comment_author] = dict_user_issue_data_init.copy(
                            )
                        user_issue_data[comment_author]['comments'] += 1

                # Get history to track the earliest date we see a user, so we fairly credit
                # them for time in the lookback period they are doing work in the Jira project
                changelog = issue.get('changelog', {}).get('histories', [])
                for history in changelog:
                    for item in history.get('items', []):
                        if item.get('field') == 'assignee':
                            history_date = datetime.strptime(
                                history['created'], '%Y-%m-%dT%H:%M:%S.%f%z').replace(tzinfo=None)
                            assignee = item.get('toString', '')
                            if assignee:
                                if assignee not in user_issue_data:
                                    user_issue_data[assignee] = dict_user_issue_data_init.copy(
                                    )
                                if (history_date < user_issue_data[assignee]['first_seen']):
                                    user_issue_data[assignee]['first_seen'] = history_date

            total_issues = data.get('total', 0)
            start_at += len(issues)
        else:
            print(
                f'Failed to fetch data: {response.status_code} - {response.text}')
            break

    # Get the earliest date we saw the user
    # those users starting after since_date have their denominator changed for per_day stats
    user_earliest_assignment_dates: Dict[str, datetime] = get_earliest_assignment_date(
        JIRA_PROJECT, since_date)

    # Calculate average open days and prepare the result
    result = []
    for user, data in user_issue_data.items():
        avg_days = data['total_days'] / \
            data['completed_issues'] if data['completed_issues'] > 0 else 0

        business_days_lookback: int = get_business_days(
            since_date, datetime.now())
        business_days_on_project: int = get_business_days(
            user_earliest_assignment_dates[user], datetime.now())
        result.append({
            'display_name': user,
            'first_seen': data['first_seen'],
            'avg_days': round(avg_days, 2),
            'created_issues': data['created_issues'],
            'completed_issues': data['completed_issues'],
            'comments': data['comments'],
            'days_on_team_in_lookback': business_days_lookback if business_days_lookback <= business_days_on_project else business_days_on_project
        })

    return result


def add_ntile_stats(df):
    ntile = 10  # Decile
    # df is the dataframe of contributor stats. Calc ntiles, add columns, return new df
    df['days_ntile'] = pd.qcut(
        df['avg_days'], ntile, labels=False, duplicates='drop')
    df['created_ntile'] = pd.qcut(
        df['created_issues_per_day'], ntile, labels=False, duplicates='drop')
    df['completed_ntile'] = pd.qcut(
        df['completed_issues_per_day'], ntile, labels=False, duplicates='drop',)
    df['comments_ntile'] = pd.qcut(
        df['comments_per_day'], ntile, labels=False, duplicates='drop',)
    # Not including days, for now
    cols_to_average = ['comments_ntile', 'created_ntile',
                       'completed_ntile']
    df['avg_ntile'] = df[cols_to_average].mean(axis=1)
    # df['grade'] = df['avg_ntile'].apply(convert_to_letter_grade)
    return df


def curve_scores(df, scores_column_name, curved_score_column_name):
    # Calculate the mean and standard deviation of the scores
    mean = df[scores_column_name].mean()
    std_dev = df[scores_column_name].std()

    # Calculate the Z-scores for each score
    z_scores = (df[scores_column_name] - mean) / std_dev

    # Create a normal distribution with mean 0 and standard deviation 1
    norm_dist = norm(0, 1)

    # Calculate the cumulative distribution function (CDF) for each Z-score
    cdf = norm_dist.cdf(z_scores)

    # Map the CDF values to a 0-100 range
    curved_scores = (cdf * 100).round().astype(int)

    # Update the DataFrame with the curved scores, near left side since this is important data
    df.insert(1, curved_score_column_name, curved_scores)

    return df

# get the date a user joined a project so we know what date to start evaluating their
# productivity


def get_earliest_assignment_date(jira_project, since_date) -> Dict[str, datetime]:
    start_at = 0
    max_results = 50  # Adjust as needed
    user_earliest_assignment = {}

    jql = f"project = {jira_project} AND updated >= '{since_date.strftime('%Y-%m-%d')}'"
    fields = "assignee"
    expand = "changelog"
    search_url = f"{JIRA_BASE_URL}/search"

    while True:
        params = {'jql': jql, 'startAt': start_at,
                  'maxResults': max_results, 'fields': fields, 'expand': expand}
        response = jira_request_exponential_backoff(search_url, params)

        if response is None:
            break

        data = response
        issues = data.get('issues', [])

        for issue in issues:
            changelog = issue.get('changelog', {}).get('histories', [])
            for history in changelog:
                for item in history.get('items', []):
                    if item.get('field') == 'assignee':
                        date = datetime.strptime(
                            history['created'], '%Y-%m-%dT%H:%M:%S.%f%z')
                        assignee = item.get('toString', '')
                        if assignee and (assignee not in user_earliest_assignment or date < user_earliest_assignment[assignee]):
                            user_earliest_assignment[assignee] = date

        start_at += max_results
        if start_at >= data.get('total', 0):
            break

    return user_earliest_assignment


if __name__ == "__main__":
    # Get the env settings
    months_lookback = int(os.getenv("DEFAULT_MONTHS_LOOKBACK", 3))
    if months_lookback < 1:
        months_lookback = 3

    default_lookback = int(os.getenv('DEFAULT_MONTHS_LOOKBACK', 3))
    since_date: datetime = get_date_months_ago(default_lookback)
    since_date_str: str = since_date.strftime('%Y-%m-%d')

    completed_issues_dict = get_completed_issues_by_user(
        JIRA_PROJECT, since_date)

    df_completed_issues = pd.DataFrame(completed_issues_dict)
    # Rename 'display_name' to 'name' to match the existing DataFrame
    df_completed_issues.rename(columns={'display_name': 'name'}, inplace=True)

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

    # Remove any rows where there are no completed issues, since that is the foundational statistic
    # I'm seeing Github return PR comments from people who were not involved in the lookback
    # period. I haven't diagnosed this. This is a hacky way to get rid of them.
    # Obvy, if PRs and commits are zero, so are changed_lines.
    df = df_completed_issues[~(df_completed_issues['completed_issues'] == 0)]
    df = add_ntile_stats(df)
    df = curve_scores(df, "avg_ntile", "curved_score")

    # print(df_completed_issues)

    # Format datetime as a string without seconds or timezone
    formatted_datetime = datetime.now().strftime('%Y-%m-%d_%H-%M')
    csv_file_path = f'jirastats_{formatted_datetime}_{JIRA_PROJECT}_Since_{since_date_str}_Stats.csv'
    df.to_csv(csv_file_path, index=False)


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
