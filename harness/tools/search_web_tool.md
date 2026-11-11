Runs a web search via DuckDuckGo and returns the top results as URL, title and excerpt. The excerpt is the snippet provided by the search backend; the tool does not visit the URL.

## PRE-EXECUTION STEPS
Before searching, please follow these steps:
1. Query Formulation:
   - Write a focused, keyword-rich query. Match the phrasing real documentation or articles would use (e.g. `pytorch transformer api 2026`, not `how do I use the transformer api in pytorch`).
   - Prefer this tool over `ShellCommandTool` with `curl`/`wget` for open-ended lookups — it returns structured results and does not require knowing the URL up front.

2. Result Sizing:
   - Use the default `num_results` for most lookups. Only raise it when you genuinely need broader coverage; the upper bound is <field>_max_num_results</field>.

## USAGE
  - `query` argument is required. Free-form search query string.
  - `num_results` argument is optional. Maximum number of results to return. Defaults to <field>_default_num_results</field>. Hard cap: <field>_max_num_results</field>.
  - `timeout` argument is optional. Per-search timeout in seconds. Defaults to <field>_default_timeout</field>. Hard cap: <field>_max_timeout</field>. Increase this when a previous call timed out.

## **VERY IMPORTANT**
- This tool returns whatever excerpt the search backend provides. It does NOT fetch the URL or read the page content. To read a page, use a separate fetch step.
- When `timed_out` is `true`, no results were retrieved at all. Retry with a higher `timeout` (up to <field>_max_timeout</field>) before giving up.
- An empty result list is NOT an error. Do not retry the same query repeatedly hoping for different output — reformulate the query instead.

## RETURNS
The tool returns a dict with the following keys:
  - `query` (str): the query that was executed (echoed for clarity).
  - `results` (list): list of result dicts, each with:
    - `url` (str): absolute URL of the result.
    - `title` (str): page title as reported by the search backend.
    - `excerpt` (str): snippet provided by the search backend.
  - `timed_out` (bool): `true` when the search exceeded `timeout` and zero results were aggregated.
