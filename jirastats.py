# jirastats.py

import os
import time
from typing import Dict, Union, Optional, List
from requests.auth import HTTPBasicAuth
import requests
import json
import base64
from datetime import datetime, timedelta, timezone
from dateutil.relativedelta import relativedelta
import pandas as pd
import numpy as np
from numpy import busday_count
from scipy.stats import norm
from requests.models import Response
from dotenv import load_dotenv


class MyJira:
    """
    A class that encapsulates functionality for interacting with a Jira instance,
    loading environment variables, querying issues, and generating Jira-based statistics.
    """

    CONFIG = {}

    def __init__(self):
        """
        Initializes the MyJira class by loading environment variables and
        populating the CONFIG dictionary with Jira-related settings.
        """
        load_dotenv()
        self.CONFIG["JIRA_API_TOKEN"] = os.getenv("JIRA_API_TOKEN")
        self.CONFIG["JIRA_CO_URL"] = os.getenv("JIRA_CO_URL")
        self.CONFIG["JIRA_USER"] = os.getenv("JIRA_USER")
        self.CONFIG["JIRA_PROJECT"] = os.getenv("JIRA_PROJECT")
        self.CONFIG["DEFAULT_MONTHS_LOOKBACK"] = int(
            os.getenv("DEFAULT_MONTHS_LOOKBACK", 3))
        self.CONFIG["JIRA_BASE_URL"] = f"https://{self.CONFIG['JIRA_CO_URL']}.atlassian.net/rest/api/3"
        self.max_page_size = 100

    def get_business_days(self, start_date: datetime, end_date: datetime) -> int:
        """
        Calculate the number of business days (weekdays) between two datetime objects.

        Args:
            start_date (datetime): The start date (inclusive).
            end_date (datetime): The end date (exclusive).

        Returns:
            int: The number of business days within the specified date range.
        """
        # Convert datetime to date if they are in datetime format
        start_date = start_date.date() if isinstance(
            start_date, datetime) else start_date
        end_date = end_date.date() if isinstance(end_date, datetime) else end_date

        # Count business days using numpy's busday_count
        return busday_count(start_date, end_date)

    def get_days(self, start: datetime, end: datetime) -> float:
        """
        Calculate the total number of days (including fractional) between two datetime objects.

        Args:
            start (datetime): The start datetime.
            end (datetime): The end datetime.

        Returns:
            float: The number of days (with fractions) between start and end.
        """
        time_difference = end - start
        days = time_difference / timedelta(days=1)
        return round(days, 2)

    def get_date_months_ago(self, months_ago: int) -> datetime:
        """
        Calculate the datetime object for a date that is a specified number of months in the past.

        Args:
            months_ago (int): The number of months to go back from the current date.

        Returns:
            datetime: The resulting date/time that is `months_ago` months before now.
        """
        current_date = datetime.now()
        date_months_ago = current_date - relativedelta(months=months_ago)
        return date_months_ago

    def encode_key(self, user_email: str, api_key: str) -> str:
        """
        Base64-encode a string composed of the `user_email` and `api_key`.

        Args:
            user_email (str): The email/username for Jira.
            api_key (str): The Jira API key/token.

        Returns:
            str: A base64-encoded string suitable for HTTP Basic Authentication.
        """
        string_to_encode = f"{user_email}:{api_key}"
        encoded_bytes = base64.b64encode(string_to_encode.encode("utf-8"))
        encoded_string = str(encoded_bytes, "utf-8")
        return encoded_string

    def sleep_until_ratelimit_reset_time(self, reset_epoch_time: int):
        """
        Sleep until the rate-limit reset time is reached, based on the given Unix epoch timestamp.

        Args:
            reset_epoch_time (int): The Unix epoch time when the rate limit resets.
        """
        reset_time = datetime.utcfromtimestamp(reset_epoch_time)
        now = datetime.utcnow()
        time_diff = reset_time - now

        if time_diff.total_seconds() < 0:
            print("\tNo sleep required. The rate limit reset time has already passed.")
        else:
            time_diff = timedelta(seconds=int(time_diff.total_seconds()))
            print(f"\tSleeping until rate limit reset: {time_diff}")
            time.sleep(time_diff.total_seconds())

    def check_API_rate_limit(self, response: Response) -> bool:
        """
        Check if the API rate limit has been exceeded and sleep until the reset time if necessary.

        Args:
            response (Response): The HTTP response object from a Jira API request.

        Returns:
            bool: True if the status code indicates a rate-limit exceedance, False otherwise.
        """
        if response.status_code == 403 and 'X-Ratelimit-Remaining' in response.headers:
            if int(response.headers['X-Ratelimit-Remaining']) == 0:
                print(
                    f"\t403 forbidden response header shows X-Ratelimit-Remaining at {response.headers['X-Ratelimit-Remaining']} requests."
                )
                self.sleep_until_ratelimit_reset_time(
                    int(response.headers['X-RateLimit-Reset']))
        return (response.status_code == 403 and 'X-Ratelimit-Remaining' in response.headers)

    def jira_request_exponential_backoff(self, url: str, params: Dict = None) -> Optional[Dict]:
        """
        Send a GET request to a given Jira REST API endpoint with exponential backoff on specific errors.

        Args:
            url (str): The Jira API endpoint URL.
            params (Dict, optional): Additional query parameters for the request.

        Returns:
            Optional[Dict]: The JSON response from the Jira API, or None if the request ultimately fails.
        """
        exponential_backoff_retry_delays_list: List[int] = [
            1, 2, 4, 8, 16, 32, 64]
        headers = {
            "Authorization": f"Basic {self.CONFIG['JIRA_API_TOKEN']}",
            "Content-Type": "application/json"
        }

        retry: bool = False
        response: Optional[Response] = None
        retry_url: Optional[str] = None

        # Initial request attempt
        try:
            response = requests.get(
                url,
                auth=HTTPBasicAuth(
                    self.CONFIG['JIRA_USER'], self.CONFIG['JIRA_API_TOKEN']),
                headers=headers,
                params=params
            )
        except requests.exceptions.Timeout:
            print("Initial request timed out.")
            retry = True

        # If first attempt fails or returns an unexpected status code, apply exponential backoff.
        if retry or (response is not None and response.status_code != 200):
            if response and response.status_code == 422 and response.reason == "Unprocessable Entity":
                dict_error: Dict[str, any] = json.loads(response.text)
                print(
                    f"Skipping: {response.status_code} {response.reason} for url {url}\n"
                    f"\t{dict_error['message']}\n"
                    f"\t{dict_error['errors'][0]['message']}"
                )
            elif retry or (response and (response.status_code == 202 or response.status_code == 403)):
                for retry_attempt_delay in exponential_backoff_retry_delays_list:
                    if response and 'Location' in response.headers:
                        retry_url = response.headers.get('Location')
                    if response and 'Retry-After' in response.headers:
                        retry_attempt_delay = int(
                            response.headers.get('Retry-After'))

                    time.sleep(retry_attempt_delay)
                    retry_response_url: str = retry_url if retry_url else url
                    print(
                        f"Retrying request for {retry_response_url} after {retry_attempt_delay} sec "
                        f"due to {response.status_code if response else 'No Response'} response"
                    )
                    self.check_API_rate_limit(response)

                    try:
                        response = requests.get(
                            retry_response_url,
                            headers=headers,
                            params=params
                        )
                    except requests.exceptions.Timeout:
                        print(
                            f"Retry request timed out. Retrying in {retry_attempt_delay} seconds."
                        )
                        continue

                    if response.status_code == 200:
                        break
                    else:
                        print(
                            f"\tRetried request and still got bad response status code: {response.status_code}"
                        )

        # Final check to see if we got a successful response
        if response and response.status_code == 200:
            return response.json()
        else:
            if response:
                self.check_API_rate_limit(response)
                print(
                    f"Retries exhausted. Giving up. Status code: {response.status_code}"
                )
            else:
                print("No valid response received.")
            return None

    def add_dict_to_dataframe(
        self,
        data_dict: Dict[str, int],
        df: pd.DataFrame,
        col_name_key: str,
        col_name_value: str
    ) -> pd.DataFrame:
        """
        Convert a dictionary into a DataFrame and append it to an existing DataFrame.

        Args:
            data_dict (Dict[str, int]): The dictionary to be added.
            df (pd.DataFrame): The DataFrame to which the dictionary data is appended.
            col_name_key (str): The name to use for the dictionary keys column in the new DataFrame.
            col_name_value (str): The name to use for the dictionary values column in the new DataFrame.

        Returns:
            pd.DataFrame: The concatenated DataFrame with the new dictionary data appended.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("df must be a pandas DataFrame")

        new_data = pd.DataFrame(list(data_dict.items()), columns=[
                                col_name_key, col_name_value])
        new_data = pd.concat([df, new_data], ignore_index=True)
        return new_data

    def calculate_avg_days_open(self, issues_data: List[Dict], df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the average days each user takes to resolve assigned issues, and store the result in a DataFrame.

        Args:
            issues_data (List[Dict]): A list of Jira issues containing 'assignee', 'fields' with 'created' and 'resolutiondate'.
            df (pd.DataFrame): A DataFrame that will be updated with the 'avg_days_open' column keyed by user 'name'.

        Returns:
            pd.DataFrame: The updated DataFrame with average open days assigned to each user in 'avg_days_open'.
        """
        user_open_days = {}

        for issue in issues_data:
            assignee = issue['assignee']
            created = issue.get('fields', {}).get('created')
            resolution_date = issue.get('fields', {}).get('resolutiondate')

            if not assignee or not created or not resolution_date:
                continue

            assigned_date = datetime.strptime(
                created, '%Y-%m-%dT%H:%M:%S.%f%z')
            completed_date = datetime.strptime(
                resolution_date, '%Y-%m-%dT%H:%M:%S.%f%z')
            open_days: float = self.get_days(assigned_date, completed_date)

            if assignee in user_open_days:
                user_open_days[assignee].append(open_days)
            else:
                user_open_days[assignee] = [open_days]

        for user, days in user_open_days.items():
            avg_days = np.mean(days)
            df.loc[df['name'] == user, 'avg_days_open'] = avg_days

        return df

    def get_earliest_assignment_date(self, jira_project: str, since_date: datetime) -> Dict[str, datetime]:
        """
        Find the earliest date a user was assigned to an issue in a given Jira project within a specified lookback.

        Args:
            jira_project (str): The key of the Jira project (e.g., "TEST").
            since_date (datetime): The earliest date to consider for updates/assignments.

        Returns:
            Dict[str, datetime]: A mapping of assignee displayName to the earliest assignment datetime observed.
        """
        start_at = 0
        max_results = 50
        user_earliest_assignment = {}

        jql = f"project = {jira_project} AND updated >= '{since_date.strftime('%Y-%m-%d')}'"
        fields = "assignee"
        expand = "changelog"
        search_url = f"{self.CONFIG['JIRA_BASE_URL']}/search"

        while True:
            params = {
                'jql': jql,
                'startAt': start_at,
                'maxResults': max_results,
                'fields': fields,
                'expand': expand
            }
            response = self.jira_request_exponential_backoff(
                search_url, params)
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

            total_issues = data.get('total', 0)
            start_at += max_results
            if start_at >= total_issues:
                break

        return user_earliest_assignment

    def get_completed_issues_by_user(self, jira_project: str, time_since: datetime) -> List[Dict[str, float]]:
        """
        Retrieve issues in a given project that are either Completed or In Progress, and gather
        statistics on how many issues each user created, completed, and commented on, as well as
        the total number of days those issues remained open.

        Args:
            jira_project (str): The Jira project key (e.g., "TEST").
            time_since (datetime): The earliest date/time to consider in the Jira issue search.

        Returns:
            List[Dict[str, float]]: A list of dictionaries where each dict represents user stats such as:
                {
                  'display_name': str,
                  'first_seen': datetime,
                  'avg_days': float,
                  'created_issues': int,
                  'completed_issues': int,
                  'comments': int,
                  'days_on_team_in_lookback': int
                }
        """
        start_at: int = 0
        total_issues: Optional[int] = None
        issue_counter: int = 0
        user_issue_data: Dict[str, Dict[str, Union[int, float]]] = {}

        # Default structure for a new user
        dict_user_issue_data_init: Dict[str, Union[int, float]] = {
            'created_issues': 0,
            'completed_issues': 0,
            'comments': 0,
            'total_days': 0.0,
            'first_seen': time_since.timestamp()
        }

        jql: str = (
            f"project = {jira_project} AND (status = 'Completed' OR status = 'In Progress') "
            f"AND (updated >= '{time_since.strftime('%Y-%m-%d %H:%M')}' "
            f"OR resolutiondate >= '{time_since.strftime('%Y-%m-%d %H:%M')}')"
        )
        fields: str = "creator,assignee,resolutiondate,created,comment,status"
        expand: str = "changelog"
        search_url: str = f"{self.CONFIG['JIRA_BASE_URL']}/search"

        while total_issues is None or start_at < total_issues:
            params: Dict = {
                'jql': jql,
                'startAt': start_at,
                'maxResults': self.max_page_size,
                'fields': fields,
                'expand': expand
            }
            response = self.jira_request_exponential_backoff(
                search_url, params)
            if response:
                issues_data = response.get('issues', [])
                total_issues = response.get('total', 0)

                for issue in issues_data:
                    issue_counter += 1
                    print(f"{issue_counter} issues processed",
                          end='\r', flush=True)

                    # Count created issues by creator
                    creator = issue['fields'].get('creator')
                    if creator:
                        creator_username: str = creator['displayName']
                        if creator_username not in user_issue_data:
                            user_issue_data[creator_username] = dict_user_issue_data_init.copy(
                            )
                        user_issue_data[creator_username]['created_issues'] += 1

                    # Count completed issues by assignee
                    assignee = issue['fields'].get('assignee')
                    status = issue['fields'].get('status')
                    if assignee and status and status["name"] == "Completed":
                        assignee_username: str = assignee['displayName']
                        created_date_str = issue['fields']['created']
                        resolution_date_str = issue['fields']['resolutiondate']

                        created_date = datetime.strptime(
                            created_date_str, '%Y-%m-%dT%H:%M:%S.%f%z')
                        resolution_date = datetime.strptime(
                            resolution_date_str, '%Y-%m-%dT%H:%M:%S.%f%z')
                        days_open = self.get_days(
                            created_date, resolution_date)

                        if assignee_username not in user_issue_data:
                            user_issue_data[assignee_username] = dict_user_issue_data_init.copy(
                            )

                        user_issue_data[assignee_username]['total_days'] += days_open
                        user_issue_data[assignee_username]['completed_issues'] += 1

                    # Count comment stats by commenter
                    comments = issue['fields'].get(
                        'comment', {}).get('comments', [])
                    for comment in comments:
                        comment_created_dt = datetime.strptime(
                            comment['created'], '%Y-%m-%dT%H:%M:%S.%f%z').replace(tzinfo=None)
                        if comment_created_dt >= time_since:
                            comment_author: str = comment['author']['displayName']
                            if comment_author not in user_issue_data:
                                user_issue_data[comment_author] = dict_user_issue_data_init.copy(
                                )
                            user_issue_data[comment_author]['comments'] += 1

                    # Track earliest assignment date from changelog
                    changelog = issue.get('changelog', {}).get('histories', [])
                    for history in changelog:
                        for item in history.get('items', []):
                            if item.get('field') == 'assignee':
                                history_date = datetime.strptime(
                                    history['created'], '%Y-%m-%dT%H:%M:%S.%f%z').replace(tzinfo=None)
                                assignee_changed_to = item.get('toString', '')
                                if assignee_changed_to:
                                    if assignee_changed_to not in user_issue_data:
                                        user_issue_data[assignee_changed_to] = dict_user_issue_data_init.copy(
                                        )
                                    if history_date.timestamp() < user_issue_data[assignee_changed_to]['first_seen']:
                                        user_issue_data[assignee_changed_to]['first_seen'] = history_date.timestamp(
                                        )

                start_at += len(issues_data)
            else:
                print('Failed to fetch data from Jira.')
                break

        # Resolve earliest assignment date from the project
        user_earliest_assignment_dates = self.get_earliest_assignment_date(
            jira_project, time_since)

        result = []
        for user, data in user_issue_data.items():
            avg_days = data['total_days'] / \
                data['completed_issues'] if data['completed_issues'] > 0 else 0.0
            business_days_lookback = self.get_business_days(
                time_since, datetime.now())

            # If a user was assigned after the lookback began, adjust days on team
            if user in user_earliest_assignment_dates:
                business_days_on_project = self.get_business_days(
                    user_earliest_assignment_dates[user], datetime.now())
            else:
                business_days_on_project = business_days_lookback

            effective_days = business_days_lookback if business_days_lookback <= business_days_on_project else business_days_on_project

            result.append({
                'display_name': user,
                'first_seen': datetime.fromtimestamp(data['first_seen']),
                'avg_days': round(avg_days, 2),
                'created_issues': int(data['created_issues']),
                'completed_issues': int(data['completed_issues']),
                'comments': int(data['comments']),
                'days_on_team_in_lookback': effective_days
            })

        return result

    def get_all_issues_for_project(self, project_key: str, allowed_statuses: Optional[List[str]] = None) -> List[Dict]:
        """
        Retrieves all Jira issues for the specified project within
        the DEFAULT_MONTHS_LOOKBACK window set in the environment variables.
        Optionally, filter by one or more statuses.

        Args:
            project_key (str): The key (ID) of the Jira project, e.g., "TEST".
            allowed_statuses (List[str], optional): A list of Jira status names to filter on.
                If None, no status filter is applied (all issues returned).

        Returns:
            List[Dict]: A list of dictionaries representing Jira issues, each containing:
                        {
                            'key': str,
                            'summary': str,
                            'status': str,
                            'description': str,
                            'assignee': str,
                            'creator': str
                        }
        """
        months_lookback = self.CONFIG["DEFAULT_MONTHS_LOOKBACK"]
        since_date = self.get_date_months_ago(months_lookback)
        since_date_str = since_date.strftime('%Y-%m-%d')

        start_at = 0
        max_results = 50
        issues_list = []

        # Base JQL to filter by project and updated date
        jql = f"project = {project_key} AND updated >= '{since_date_str}'"

        # If we have a status filter, add it to the JQL clause
        if allowed_statuses:
            # Create a parenthesized OR condition for multiple statuses
            status_conditions = " OR ".join(
                [f"status = \"{status}\"" for status in allowed_statuses])
            jql += f" AND ({status_conditions})"

        fields = "summary,status,description,assignee,creator"
        search_url = f"{self.CONFIG['JIRA_BASE_URL']}/search"

        while True:
            params = {
                'jql': jql,
                'startAt': start_at,
                'maxResults': max_results,
                'fields': fields
            }
            response_data = self.jira_request_exponential_backoff(
                search_url, params)
            if not response_data:
                break

            issues = response_data.get('issues', [])
            for issue in issues:
                fields_data = issue.get('fields', {})
                issues_list.append({
                    'key': issue.get('key'),
                    'summary': fields_data.get('summary'),
                    'status': fields_data.get('status', {}).get('name'),
                    'description': fields_data.get('description'),
                    'assignee': fields_data.get('assignee', {}).get('displayName') if fields_data.get('assignee') else None,
                    'creator': fields_data.get('creator', {}).get('displayName') if fields_data.get('creator') else None
                })

            total_issues = response_data.get('total', 0)
            start_at += len(issues)
            if start_at >= total_issues:
                break

        return issues_list

    def test_get_all_issues_for_project(self):
        """
        Test method to retrieve all issues for the Jira project specified in CONFIG["JIRA_PROJECT"].
        Prints out the total number of issues found and the first five issue dictionaries.
        """
        project_name = self.CONFIG["JIRA_PROJECT"]
        if not project_name:
            print("No project name set in environment variable JIRA_PROJECT.")
            return

        issues = self.get_all_issues_for_project(project_name)
        print(f"Found {len(issues)} issues for project {project_name}.")
        for issue in issues[:5]:
            print(issue)

    def add_ntile_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate and add decile-based ntile metrics to the dataframe,
        including days_ntile, created_ntile, completed_ntile, and comments_ntile.

        Args:
            df (pd.DataFrame): A DataFrame containing columns avg_days, created_issues_per_day,
                               completed_issues_per_day, comments_per_day.

        Returns:
            pd.DataFrame: The same DataFrame with additional ntile columns and an 'avg_ntile' column.
        """
        ntile = 10  # Decile
        df['days_ntile'] = pd.qcut(
            df['avg_days'], ntile, labels=False, duplicates='drop')
        df['created_ntile'] = pd.qcut(
            df['created_issues_per_day'], ntile, labels=False, duplicates='drop')
        df['completed_ntile'] = pd.qcut(
            df['completed_issues_per_day'], ntile, labels=False, duplicates='drop')
        df['comments_ntile'] = pd.qcut(
            df['comments_per_day'], ntile, labels=False, duplicates='drop')

        # Combine relevant ntile columns
        cols_to_average = ['comments_ntile',
                           'created_ntile', 'completed_ntile']
        df['avg_ntile'] = df[cols_to_average].mean(axis=1)
        return df

    def curve_scores(self, df: pd.DataFrame, scores_column_name: str, curved_score_column_name: str) -> pd.DataFrame:
        """
        Curve scores in the specified column using the Normal distribution to map each value to a 0-100 range.

        Args:
            df (pd.DataFrame): The DataFrame containing the scores to curve.
            scores_column_name (str): Name of the column in df to be curved.
            curved_score_column_name (str): Name of the new column to store the curved scores.

        Returns:
            pd.DataFrame: The DataFrame with an inserted column for the curved scores.
        """
        mean_val = df[scores_column_name].mean()
        std_dev = df[scores_column_name].std()

        z_scores = (df[scores_column_name] - mean_val) / std_dev

        norm_dist = norm(0, 1)
        cdf = norm_dist.cdf(z_scores)

        curved_scores = (cdf * 100).round().astype(int)
        df.insert(1, curved_score_column_name, curved_scores)

        return df

    def get_jira_stats(self) -> pd.DataFrame:
        """
        Generate a DataFrame containing Jira-based statistics, including average days to close,
        number of created issues, completed issues, and comments per day. Data is filtered within
        the lookback set by DEFAULT_MONTHS_LOOKBACK. The resulting DataFrame is saved to a CSV file.

        Returns:
            pd.DataFrame: The DataFrame containing user-based statistics and advanced metrics like ntiles.
        """
        months_lookback = self.CONFIG["DEFAULT_MONTHS_LOOKBACK"]
        if months_lookback < 1:
            months_lookback = 3

        since_date: datetime = self.get_date_months_ago(months_lookback)
        since_date_str: str = since_date.strftime('%Y-%m-%d')

        completed_issues_dict = self.get_completed_issues_by_user(
            self.CONFIG["JIRA_PROJECT"], since_date
        )

        df_completed_issues = pd.DataFrame(completed_issues_dict)
        df_completed_issues.rename(
            columns={'display_name': 'name'}, inplace=True)
        df_completed_issues.fillna({'assigned_issues': 0, 'avg_days': 0, 'completed_issues': 0,
                                    'comments': 0}, inplace=True)
        df_completed_issues['created_issues'] = df_completed_issues['created_issues'].astype(
            int)
        df_completed_issues['completed_issues'] = df_completed_issues['completed_issues'].astype(
            int)

        business_days = self.get_business_days(since_date, datetime.now())

        df_completed_issues['created_issues_per_day'] = (
            df_completed_issues['created_issues'] / business_days
        ).round(2)
        df_completed_issues['completed_issues_per_day'] = (
            df_completed_issues['completed_issues'] / business_days
        ).round(2)
        df_completed_issues['comments_per_day'] = (
            df_completed_issues['comments'] / business_days
        ).round(2)

        # Filter out users with zero completed issues
        df = df_completed_issues[~(
            df_completed_issues['completed_issues'] == 0)]
        df = self.add_ntile_stats(df)
        df = self.curve_scores(df, "avg_ntile", "curved_score")

        formatted_datetime = datetime.now().strftime('%Y-%m-%d_%H-%M')
        csv_file_path = f'jirastats_{formatted_datetime}_{self.CONFIG["JIRA_PROJECT"]}_Since_{since_date_str}_Stats.csv'
        df.to_csv(csv_file_path, index=False)

        return df

    def build_human_readable_issue_text(self, issue: Dict) -> str:
        """
        Convert a single Jira issue dictionary into a human-readable sentence.

        Args:
            issue (Dict): A Jira issue, presumably containing keys:
                'key', 'summary', 'status', 'description', 'assignee', 'creator'.

        Returns:
            str: A single human-readable sentence summarizing the issue.
        """

        # 1. Extract main fields.
        issue_key = issue.get("key", "UNKNOWN-KEY")
        summary = issue.get("summary", "No Summary Provided")
        status = issue.get("status", "No Status Provided")

        # 2. Extract the nested 'description' which follows the Atlassian doc structure.
        description_data = issue.get("description", {})
        description_text = self._parse_prosemirror_doc(description_data)
        if not description_text.strip():
            description_text = "No detailed description provided."

        # 3. Build a single-line or multi-line string that references each piece of data:
        #    You can adjust formatting to your preference.
        readable_str = (
            f"The Jira issue {issue_key} has summary: '{summary}', "
            f"status: '{status}', and the following detail:\n\n{description_text}"
        )

        return readable_str

    def _parse_prosemirror_doc(self, doc: Dict) -> str:
        """
        A helper function to recursively parse the ProseMirror-based 'description' field
        returned by Jira. This will collect text from all nested paragraphs, bullet points, etc.

        Args:
            doc (Dict): The root dictionary for the ProseMirror document.

        Returns:
            str: A simplified text rendering of the doc content.
        """

        # If it's not a dict or has no "content", there's nothing to parse
        if not isinstance(doc, dict):
            return ""
        content = doc.get("content", [])
        if not content:
            # Sometimes doc itself might be a block with type, content
            # We'll parse it if it's structured the same as children
            return ""

        # We'll traverse each content node and pick out text from paragraphs, bullet lists, etc.
        texts = []
        for node in content:
            node_type = node.get("type", "")
            node_text = ""

            # If this node has children, we parse them recursively
            if "content" in node:
                node_text = self._parse_prosemirror_doc(node)

            # If this node is 'text', gather up text
            elif node_type == "text":
                node_text = node.get("text", "")

            # If we have sub-lists or paragraphs, parse them recursively
            # or treat them as line breaks or bullet points, etc.
            # But we can rely on the recursion above to gather everything.

            # For readability, let's place each top-level node's text on a new line
            if node_text.strip():
                texts.append(node_text.strip())

        # Join with newlines for readability
        return "\n".join(texts)

    def get_human_readable_issues_for_project(self, project_key: str) -> List[str]:
        """
        Retrieve all issues for a project and convert each issue into a 
        single, human-readable string.

        Args:
            project_key (str): The Jira project key (e.g., "PAYHELP").

        Returns:
            List[str]: A list of human-readable strings summarizing each issue.
        """
        issues = self.get_all_issues_for_project(project_key)
        summaries = []
        for issue in issues:
            summary_text = self.build_human_readable_issue_text(issue)
            summaries.append(summary_text)
        return summaries

# ------------------ TEST / MAIN-EXECUTION CODE ------------------ #


def test_get_all_issues_for_project():
    """
    A standalone test function to verify fetching of Jira issues for the project defined in .env.
    """
    jira_instance = MyJira()
    jira_instance.test_get_all_issues_for_project()


def get_jira_stats():
    """
    A standalone function to generate a DataFrame of Jira stats and print the result.
    """
    jira_instance = MyJira()
    df = jira_instance.get_jira_stats()
    print(df)


if __name__ == "__main__":
    jira_instance = MyJira()
    project_key = jira_instance.CONFIG["JIRA_PROJECT"]
    human_readable_list = jira_instance.get_human_readable_issues_for_project(
        project_key)

    for readable_issue in human_readable_list:
        print(readable_issue)
        print("---")
    # test_get_all_issues_for_project()
    # get_jira_stats()
    print("Done.")
