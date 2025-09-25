"""
Personal Finance Dashboard built with Streamlit
================================================

This app projects daily cash flow from today through a
user‑selected end date (default is the end of 2025) based on
user‑provided income and expenses.  It also allows the user to
define one‑time targets (birthdays, holidays, attorney fees, etc.)
and a debt payoff date.  The dashboard displays key metrics,
charts, a weekly plan table, and optional AI‑powered advice.

The code below is heavily commented so that non‑programmers can
understand the logic and extend it later.  All inputs live in
Streamlit’s sidebar and defaults are pre‑filled from the project
specification.

Note: to deploy this on Streamlit Community Cloud you simply need
this file, the accompanying ``requirements.txt``, and the
``README.md``.  See the README for step‑by‑step instructions.
"""

import json
import math
from datetime import date, datetime, timedelta
from typing import List, Dict, Tuple

import pandas as pd
import streamlit as st
from dateutil.relativedelta import relativedelta

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Personal Finance Dashboard",
    layout="wide",
    page_icon="💸",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def parse_targets(json_str: str) -> List[Dict]:
    """Parse the JSON string of targets into a list of dictionaries.

    Each target should have the keys ``date`` (YYYY‑MM‑DD), ``name`` and
    ``amount``.  If the JSON fails to parse, an empty list is returned
    and an error is displayed on the page.
    """
    try:
        targets = json.loads(json_str)
        # Ensure each target has the required fields
        cleaned = []
        for t in targets:
            if not all(k in t for k in ("date", "name", "amount")):
                continue
            cleaned.append(
                {
                    "date": t["date"],
                    "name": t["name"],
                    "amount": float(t["amount"]),
                }
            )
        return cleaned
    except Exception as e:
        st.error(f"Unable to parse targets JSON: {e}")
        return []


@st.cache_data(show_spinner=False)
def compute_daily_schedule(
    start_date: date,
    end_date: date,
    daily_income: float,
    daily_living: float,
    housing_weekly: float,
    phone_monthly: float,
    car_monthly: float,
    storage_monthly: float,
    targets: List[Dict],
    debt_amount: float,
    debt_date: date,
) -> pd.DataFrame:
    """Generate a DataFrame with one row per day between ``start_date`` and
    ``end_date`` (inclusive) containing projected income and expense
    categories, daily net cash and cumulative balance.

    This schedule treats all income and fixed expenses as occurring on
    their respective due dates: income arrives daily, housing is paid on
    Mondays, monthly bills hit on fixed calendar days and one‑time
    targets hit on their scheduled date.  Debt is paid on the selected
    ``debt_date``.

    The resulting DataFrame has the following columns:

    - ``date``: the calendar date
    - ``income``: income received that day
    - ``daily_living``: daily living expense for that day
    - ``weekly_housing``: housing payment on Mondays, else zero
    - ``phone``, ``car``, ``storage``: monthly bills when their
      respective day matches the calendar day
    - ``target_payment``: sum of one‑time targets due on that date
    - ``debt_payment``: the debt payoff amount on the chosen date
    - ``expenses``: sum of the above expense categories
    - ``net``: income minus expenses
    - ``balance``: cumulative sum of net through the schedule
    """
    # Create a date range for the planning horizon
    dates = pd.date_range(start_date, end_date, freq="D")
    df = pd.DataFrame({"date": dates})

    # Constant daily income and living expenses
    df["income"] = daily_income
    df["daily_living"] = daily_living

    # Weekly housing: pay on Monday (weekday == 0)
    df["weekly_housing"] = df["date"].apply(
        lambda d: housing_weekly if d.weekday() == 0 else 0
    )

    # Monthly bills: charged on specific calendar days
    df["phone"] = df["date"].apply(
        lambda d: phone_monthly if d.day == 17 else 0
    )
    df["car"] = df["date"].apply(
        lambda d: car_monthly if d.day == 15 else 0
    )
    df["storage"] = df["date"].apply(
        lambda d: storage_monthly if d.day == 18 else 0
    )

    # One‑time targets: initialize all zeros then add amounts on their dates
    df["target_payment"] = 0.0
    for tgt in targets:
        try:
            tgt_date = datetime.strptime(tgt["date"], "%Y-%m-%d").date()
        except ValueError:
            continue
        # Add the target amount to the matching date
        df.loc[df["date"] == pd.Timestamp(tgt_date), "target_payment"] += tgt[
            "amount"
        ]

    # Debt payment on selected date
    df["debt_payment"] = df["date"].apply(
        lambda d: debt_amount if d.date() == debt_date else 0
    )

    # Total expenses per day (sum of all expense categories)
    df["expenses"] = (
        df["daily_living"]
        + df["weekly_housing"]
        + df["phone"]
        + df["car"]
        + df["storage"]
        + df["target_payment"]
        + df["debt_payment"]
    )

    # Net cash flow per day
    df["net"] = df["income"] - df["expenses"]

    # Running balance (cumulative sum of net)
    df["balance"] = df["net"].cumsum()

    return df


@st.cache_data(show_spinner=False)
def compute_weekly_plan(
    daily_df: pd.DataFrame,
    start_date: date,
    targets: List[Dict],
    debt_amount: float,
    debt_date: date,
    housing_weekly: float,
    phone_monthly: float,
    car_monthly: float,
    storage_monthly: float,
    daily_living: float,
    daily_income: float,
) -> Tuple[pd.DataFrame, Dict[str, Dict], List[Dict]]:
    """Aggregate the daily schedule into weekly buckets and compute
    recommended sinking fund contributions for each target.

    Returns a tuple of (weekly_df, on_track_map, target_recs)

    - ``weekly_df`` contains one row per week with the following columns:
        - ``week_start``: the Monday of the week
        - ``Income``: sum of income for the week
        - ``Fixed Bills``: housing + monthly bills
        - ``Daily Living``: sum of daily living costs
        - ``Target Set-Aside``: recommended sinking fund contribution for that week
        - ``Debt Payment``: debt payment in that week
        - ``Net``: Income − (Fixed Bills + Daily Living + Target Set-Aside + Debt Payment)
        - ``Running Balance``: cumulative sum of ``Net`` across the weeks
        - ``On-Track?``: boolean flag indicating whether the daily schedule shows
          a non‑negative balance on or before each target’s due date
    - ``on_track_map`` is keyed by target name and holds a dict with the
      fields ``on_track``, ``balance_before_date`` and ``due_date``
    - ``target_recs`` is a list of targets augmented with ``weeks_remaining``
      and ``weekly_contrib`` fields for display
    """
    # Compute the start of the current week (Monday).  This ensures that
    # contributions align neatly to calendar weeks.
    start_week_start = start_date - timedelta(days=start_date.weekday())

    # Prepare a mapping of week_start dates to recommended contributions
    contrib_map: Dict[date, float] = {}
    target_recs: List[Dict] = []

    # For each target calculate the number of whole weeks left and the
    # contribution required per week.  Use ceil() to ensure that we fully
    # fund the target even if the remaining period is not an exact multiple
    # of seven days.
    for tgt in targets:
        try:
            due_date = datetime.strptime(tgt["date"], "%Y-%m-%d").date()
        except Exception:
            continue
        days_left = max((due_date - start_date).days, 0)
        # Always allocate at least one week to avoid division by zero
        weeks_remaining = max(math.ceil(days_left / 7.0), 1)
        weekly_contrib = tgt["amount"] / weeks_remaining
        target_recs.append(
            {
                **tgt,
                "weeks_remaining": weeks_remaining,
                "weekly_contrib": weekly_contrib,
            }
        )
        # Allocate contributions to each week up to the due date
        for i in range(weeks_remaining):
            wk_start = start_week_start + timedelta(weeks=i)
            if wk_start not in contrib_map:
                contrib_map[wk_start] = 0.0
            contrib_map[wk_start] += weekly_contrib

    # Now build the weekly DataFrame by aggregating the daily schedule
    df = daily_df.copy()
    df["week_start"] = df["date"].apply(
        lambda d: d - pd.Timedelta(days=d.weekday())
    )
    weekly = (
        df.groupby("week_start")[
            ["income", "daily_living", "weekly_housing", "phone", "car", "storage", "target_payment", "debt_payment"]
        ]
        .sum()
        .reset_index()
    )

    # Compute fixed bills (housing + monthly bills)
    weekly["Fixed Bills"] = (
        weekly["weekly_housing"]
        + weekly["phone"]
        + weekly["car"]
        + weekly["storage"]
    )

    # Daily living is already aggregated
    weekly.rename(columns={"income": "Income", "daily_living": "Daily Living"}, inplace=True)

    # Debt Payment (already aggregated)
    weekly.rename(columns={"debt_payment": "Debt Payment"}, inplace=True)

    # Prepare Target Set‑Aside column based on the contribution map
    weekly["Target Set-Aside"] = weekly["week_start"].apply(
        lambda wk: contrib_map.get(wk.date(), 0.0)
    )

    # Compute Net = Income − (Fixed Bills + Daily Living + Target Set‑Aside + Debt Payment)
    weekly["Net"] = (
        weekly["Income"]
        - (weekly["Fixed Bills"] + weekly["Daily Living"] + weekly["Target Set-Aside"] + weekly["Debt Payment"])
    )

    # Running balance across weeks based on recommended contributions
    weekly["Running Balance"] = weekly["Net"].cumsum()

    # Determine if each target is on track based on the actual daily balance
    on_track_map: Dict[str, Dict] = {}
    for tgt in targets:
        name = tgt.get("name")
        due_str = tgt.get("date")
        amount = float(tgt.get("amount", 0))
        try:
            due_date = datetime.strptime(due_str, "%Y-%m-%d").date()
        except Exception:
            continue
        # Find the row in the daily schedule for the day before the target date
        idx = daily_df[daily_df["date"] == pd.Timestamp(due_date) - pd.Timedelta(days=1)].index
        if len(idx) == 0:
            balance_before = 0.0
        else:
            balance_before = float(daily_df.loc[idx[0], "balance"])
        # We consider the plan on track if the balance before the target date
        # remains non‑negative.  Another way to check would be
        # (balance_before - amount >= 0) which ensures full coverage of the
        # target, but the spec only requires the balance itself to be non‑negative.
        on_track = balance_before >= 0
        on_track_map[name] = {
            "on_track": on_track,
            "balance_before": balance_before,
            "due_date": due_date,
        }

    # Add the on‑track flag to each row.  A week is considered on track if
    # all targets due up to and including that week are on track.  We check
    # the due_date against the end of the week.
    weekly["On-Track?"] = True
    for i, row in weekly.iterrows():
        week_end = row["week_start"].date() + timedelta(days=6)
        # All targets whose due_date <= week_end must be on track
        for tgt_name, info in on_track_map.items():
            if info["due_date"] <= week_end and not info["on_track"]:
                weekly.at[i, "On-Track?"] = False
                break

    # Select and order the columns for display
    weekly_display = weekly[
        [
            "week_start",
            "Income",
            "Fixed Bills",
            "Daily Living",
            "Target Set-Aside",
            "Debt Payment",
            "Net",
            "Running Balance",
            "On-Track?",
        ]
    ].copy()
    # Rename week_start for readability
    weekly_display.rename(columns={"week_start": "Week Start"}, inplace=True)

    return weekly_display, on_track_map, target_recs


def compute_runway_days(balance_series: pd.Series) -> int:
    """Find the number of days until the running balance becomes negative.

    Returns the number of days from the start until the balance first drops
    below zero.  If it never drops below zero, the full length of the
    series is returned.
    """
    for idx, bal in enumerate(balance_series):
        if bal < 0:
            return idx  # zero‑based index corresponds to days from start
    return len(balance_series)


def format_currency(x: float) -> str:
    """Utility to format a number as USD."""
    return f"${x:,.2f}"


def generate_ai_advice(
    api_key: str,
    daily_income: float,
    daily_living: float,
    housing_weekly: float,
    phone_monthly: float,
    car_monthly: float,
    storage_monthly: float,
    targets: List[Dict],
    debt_amount: float,
    runway_weeks: float,
) -> str:
    """Use OpenAI to generate three action items for the upcoming week.

    This function is only invoked when the user has supplied a valid API key
    and clicked the advice button.  It constructs a prompt describing the
    current financial situation and asks for three concrete, actionable
    suggestions.  If the API call fails, an error message is returned.
    """
    import openai

    openai.api_key = api_key
    # Build a summary of the user’s situation to pass into the AI
    target_desc = ", ".join(
        [f"{t['name']} (${t['amount']}) by {t['date']}" for t in targets]
    )
    system_prompt = (
        "You are an assistant helping a user manage their personal finances.\n"
        "Provide three actionable and concrete suggestions for the upcoming week.\n"
        "Suggestions can include ideas to earn extra income, reduce discretionary \n"
        "spending, renegotiate or shift due dates, or otherwise improve the user's \n"
        "cash flow.  Be concise and supportive.  Do not mention that you are an AI.\n"
    )
    user_prompt = (
        f"Current daily income is ${daily_income} and daily living expenses are ${daily_living}. "
        f"Housing is ${housing_weekly}/week, phone is ${phone_monthly} on the 17th of each month, "
        f"car insurance is ${car_monthly} on the 15th, storage is ${storage_monthly} on the 18th. "
        f"Debt to pay is ${debt_amount}. Upcoming targets: {target_desc}. "
        f"Current runway is {runway_weeks:.1f} weeks. Please provide 3 unique suggestions."
    )
    try:
        # Use ChatCompletion for better results
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=300,
            temperature=0.7,
        )
        advice = response["choices"][0]["message"]["content"].strip()
        return advice
    except Exception as e:
        return f"Error generating advice: {e}"


# ---------------------------------------------------------------------------
# Sidebar inputs
# ---------------------------------------------------------------------------
st.sidebar.title("📋 Settings")

today = date.today()

# Income and expense defaults
daily_income = st.sidebar.number_input(
    "Daily income ($)",
    min_value=0.0,
    value=500.0,
    step=10.0,
    help="Your average income per day from all sources."
)
daily_living = st.sidebar.number_input(
    "Daily living expenses ($)",
    min_value=0.0,
    value=100.0,
    step=5.0,
    help="Daily food, transportation and other living costs."
)
housing_weekly = st.sidebar.number_input(
    "Housing (weekly, $)",
    min_value=0.0,
    value=665.0,
    step=5.0,
    help="Rent or mortgage payment that occurs once per week (assumed Monday)."
)
phone_monthly = st.sidebar.number_input(
    "Phone bill (17th each month, $)",
    min_value=0.0,
    value=60.0,
    step=5.0,
)
car_monthly = st.sidebar.number_input(
    "Car insurance (15th each month, $)",
    min_value=0.0,
    value=300.0,
    step=10.0,
)
storage_monthly = st.sidebar.number_input(
    "Storage (18th each month, $)",
    min_value=0.0,
    value=146.0,
    step=5.0,
)

# Debt inputs
debt_amount = st.sidebar.number_input(
    "Current debt ($)",
    min_value=0.0,
    value=1470.0,
    step=10.0,
    help="Total amount of debt you plan to pay on the chosen debt payment date."
)
debt_date = st.sidebar.date_input(
    "Debt payment date",
    value=today + timedelta(days=30),
    min_value=today,
    help="Date on which you plan to pay off your debt."
)

# Planning horizon
end_date = st.sidebar.date_input(
    "Projection end date", value=date(2025, 12, 31), min_value=today,
    help="The dashboard models cash flow until this date."
)

# Targets input as JSON
default_targets = [
    {"date": "2025-10-06", "name": "Gifts for daughter", "amount": 500},
    {"date": "2025-10-11", "name": "Daughter’s birthday party", "amount": 500},
    {"date": "2025-10-31", "name": "Halloween", "amount": 300},
    {"date": "2025-11-24", "name": "Thanksgiving/Black Friday", "amount": 500},
    {"date": "2025-12-01", "name": "Attorney fee", "amount": 10000},
    {"date": "2025-12-08", "name": "Christmas gifts", "amount": 500},
]
targets_json_default = json.dumps(default_targets, indent=2)

targets_json_str = st.sidebar.text_area(
    "Targets (JSON)",
    value=targets_json_default,
    height=200,
    help="Edit this JSON to add, remove or modify one‑time targets. Each target must have a 'date' (YYYY‑MM‑DD), a 'name' and an 'amount'."
)
targets_list = parse_targets(targets_json_str)

# Persistence: download and upload settings
st.sidebar.markdown("**Save/Load Settings**")
settings_dict = {
    "daily_income": daily_income,
    "daily_living": daily_living,
    "housing_weekly": housing_weekly,
    "phone_monthly": phone_monthly,
    "car_monthly": car_monthly,
    "storage_monthly": storage_monthly,
    "debt_amount": debt_amount,
    "debt_date": debt_date.isoformat(),
    "end_date": end_date.isoformat(),
    "targets": targets_list,
}
settings_json = json.dumps(settings_dict, indent=2)
st.sidebar.download_button(
    "Download settings (JSON)",
    settings_json,
    file_name="finance_settings.json",
    mime="application/json",
)

uploaded_settings_file = st.sidebar.file_uploader(
    "Upload settings (JSON)", type=["json"], help="Load a previously saved settings file."
)
if uploaded_settings_file:
    try:
        uploaded_settings = json.load(uploaded_settings_file)
        # Assign the uploaded values back into st.session_state so that the UI updates
        st.session_state["daily_income"] = float(uploaded_settings.get("daily_income", daily_income))
        st.session_state["daily_living"] = float(uploaded_settings.get("daily_living", daily_living))
        st.session_state["housing_weekly"] = float(uploaded_settings.get("housing_weekly", housing_weekly))
        st.session_state["phone_monthly"] = float(uploaded_settings.get("phone_monthly", phone_monthly))
        st.session_state["car_monthly"] = float(uploaded_settings.get("car_monthly", car_monthly))
        st.session_state["storage_monthly"] = float(uploaded_settings.get("storage_monthly", storage_monthly))
        st.session_state["debt_amount"] = float(uploaded_settings.get("debt_amount", debt_amount))
        st.session_state["debt_date"] = datetime.strptime(uploaded_settings.get("debt_date", debt_date.isoformat()), "%Y-%m-%d").date()
        st.session_state["end_date"] = datetime.strptime(uploaded_settings.get("end_date", end_date.isoformat()), "%Y-%m-%d").date()
        # Recreate the targets JSON text area with the uploaded targets
        st.session_state["Targets (JSON)"] = json.dumps(uploaded_settings.get("targets", default_targets), indent=2)
        st.success("Settings loaded. Please adjust the sidebar inputs if needed.")
    except Exception as e:
        st.error(f"Failed to load settings: {e}")

# Optional AI advisor settings
st.sidebar.markdown("**AI Advisor (optional)**")
openai_key = st.sidebar.text_input(
    "OpenAI API key (optional)",
    type="password",
    help="Paste your OpenAI API key here to enable AI suggestions. Leave blank to disable."
)
generate_advice = st.sidebar.button("Generate weekly advice")

# ---------------------------------------------------------------------------
# Main dashboard
# ---------------------------------------------------------------------------

# Compute the daily schedule and weekly plan
daily_schedule = compute_daily_schedule(
    start_date=today,
    end_date=end_date,
    daily_income=daily_income,
    daily_living=daily_living,
    housing_weekly=housing_weekly,
    phone_monthly=phone_monthly,
    car_monthly=car_monthly,
    storage_monthly=storage_monthly,
    targets=targets_list,
    debt_amount=debt_amount,
    debt_date=debt_date,
)

weekly_plan, on_track_map, target_recs = compute_weekly_plan(
    daily_df=daily_schedule,
    start_date=today,
    targets=targets_list,
    debt_amount=debt_amount,
    debt_date=debt_date,
    housing_weekly=housing_weekly,
    phone_monthly=phone_monthly,
    car_monthly=car_monthly,
    storage_monthly=storage_monthly,
    daily_living=daily_living,
    daily_income=daily_income,
)

# Compute KPI values
final_balance = float(daily_schedule.iloc[-1]["balance"]) if not daily_schedule.empty else 0.0

# Net for this week (first row of weekly plan)
this_week_net = float(weekly_plan.iloc[0]["Net"]) if not weekly_plan.empty else 0.0

# Runway calculation: number of days until balance goes negative
runway_days = compute_runway_days(daily_schedule["balance"])
runway_weeks = runway_days / 7.0

# Determine next target and projected balance before it
upcoming_targets = sorted(
    [t for t in targets_list if datetime.strptime(t["date"], "%Y-%m-%d").date() >= today],
    key=lambda t: t["date"],
)
if upcoming_targets:
    next_tgt = upcoming_targets[0]
    next_tgt_date = datetime.strptime(next_tgt["date"], "%Y-%m-%d").date()
    idx = daily_schedule[daily_schedule["date"] == pd.Timestamp(next_tgt_date) - pd.Timedelta(days=1)].index
    if len(idx) > 0:
        projected_balance = float(daily_schedule.loc[idx[0], "balance"])
    else:
        projected_balance = float(daily_schedule.iloc[0]["balance"])
    next_target_label = f"{next_tgt['name']} ({next_tgt['date']})"
else:
    next_target_label = "No upcoming target"
    projected_balance = final_balance

# ---------------------------------------------------------------------------
# Layout: KPI cards
# ---------------------------------------------------------------------------

st.title("💸 Personal Finance Dashboard")
st.markdown(
    "Use the sidebar to adjust your income, expenses and targets. "
    "The dashboard below will update automatically with your changes."
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Projected end balance", format_currency(final_balance))
col2.metric("This week's net", format_currency(this_week_net))
col3.metric("Runway (weeks)", f"{runway_weeks:.1f}")
col4.metric("Next target & balance", f"{next_target_label}\n{format_currency(projected_balance)}")

# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

st.subheader("Balance over time")
st.line_chart(
    daily_schedule.set_index("date")["balance"], use_container_width=True
)

st.subheader("Weekly income vs expenses")
# Build a DataFrame for the bar chart with Income and combined expenses
bar_df = weekly_plan[["Week Start", "Income", "Fixed Bills", "Daily Living", "Target Set-Aside", "Debt Payment"]].copy()
bar_df = bar_df.set_index("Week Start")
st.bar_chart(bar_df, use_container_width=True)

# ---------------------------------------------------------------------------
# Weekly plan table
# ---------------------------------------------------------------------------

st.subheader("Weekly plan")
st.dataframe(
    weekly_plan.style.format({
        "Income": "${:,.2f}",
        "Fixed Bills": "${:,.2f}",
        "Daily Living": "${:,.2f}",
        "Target Set-Aside": "${:,.2f}",
        "Debt Payment": "${:,.2f}",
        "Net": "${:,.2f}",
        "Running Balance": "${:,.2f}",
    }),
    use_container_width=True,
)

# ---------------------------------------------------------------------------
# Targets & Sinking funds
# ---------------------------------------------------------------------------

st.subheader("Targets & sinking funds")
if target_recs:
    for tgt in target_recs:
        col_a, col_b = st.columns([2, 1])
        with col_a:
            st.markdown(f"**{tgt['name']}** – due {tgt['date']}")
            st.caption(f"Amount: {format_currency(tgt['amount'])}")
            st.caption(
                f"Recommended weekly contribution: {format_currency(tgt['weekly_contrib'])} "
                f"over {tgt['weeks_remaining']} weeks"
            )
        with col_b:
            # Show a progress bar representing the fraction of weeks elapsed
            try:
                due_date = datetime.strptime(tgt["date"], "%Y-%m-%d").date()
                total_days = (due_date - today).days
                elapsed_days = (today - today).days  # always zero on load
                # If the target date is in the past, mark as complete
                progress = 1.0 if total_days <= 0 else min(elapsed_days / total_days, 1.0)
            except Exception:
                progress = 0.0
            st.progress(progress)
            # Indicate on‑track status from daily schedule
            track_info = on_track_map.get(tgt["name"], {})
            if track_info:
                emoji = "✅" if track_info["on_track"] else "⚠️"
                st.caption(
                    f"{emoji} Projected balance before date: {format_currency(track_info['balance_before'])}"
                )
else:
    st.write("No targets defined.")

# ---------------------------------------------------------------------------
# Calendar of upcoming bills and targets
# ---------------------------------------------------------------------------

st.subheader("Upcoming bills & targets (next 90 days)")
upcoming_events: List[Dict[str, str]] = []
future_horizon = today + timedelta(days=90)
for d in pd.date_range(today, future_horizon, freq="D"):
    # Monthly bills
    if d.day == 15:
        upcoming_events.append({"Date": d.date(), "Type": "Car insurance", "Amount": format_currency(car_monthly)})
    if d.day == 17:
        upcoming_events.append({"Date": d.date(), "Type": "Phone bill", "Amount": format_currency(phone_monthly)})
    if d.day == 18:
        upcoming_events.append({"Date": d.date(), "Type": "Storage", "Amount": format_currency(storage_monthly)})
    # Housing on Mondays
    if d.weekday() == 0:
        upcoming_events.append({"Date": d.date(), "Type": "Housing", "Amount": format_currency(housing_weekly)})

    # Targets
    for tgt in targets_list:
        if tgt["date"] == d.date().isoformat():
            upcoming_events.append({"Date": d.date(), "Type": tgt["name"], "Amount": format_currency(tgt["amount"])})

if upcoming_events:
    calendar_df = pd.DataFrame(upcoming_events).sort_values(by="Date")
    calendar_df["Date"] = calendar_df["Date"].astype(str)
    st.table(calendar_df)
else:
    st.write("No upcoming bills or targets in the next 90 days.")

# ---------------------------------------------------------------------------
# AI advice section
# ---------------------------------------------------------------------------

if generate_advice:
    st.subheader("Weekly advice")
    if not openai_key:
        st.error("Please provide a valid OpenAI API key in the sidebar to enable AI advice.")
    else:
        with st.spinner("Generating advice..."):
            advice_text = generate_ai_advice(
                api_key=openai_key.strip(),
                daily_income=daily_income,
                daily_living=daily_living,
                housing_weekly=housing_weekly,
                phone_monthly=phone_monthly,
                car_monthly=car_monthly,
                storage_monthly=storage_monthly,
                targets=targets_list,
                debt_amount=debt_amount,
                runway_weeks=runway_weeks,
            )
        st.write(advice_text)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown(
    "---\n"
    "*This dashboard is a forecasting tool. All figures are projections based on the inputs provided. Actual income and expenses may vary.*"
)
