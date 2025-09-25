# Personal Finance Dashboard

This repository contains a simple Streamlit application that helps you project
your cash flow, budget for upcoming expenses and receive optional AI‑powered
tips.  No coding skills are required – just follow the setup steps below.

## Files included

| File | Purpose |
|-----|---------|
| **streamlit_app.py** | The Streamlit application. Contains the data model, UI components and optional OpenAI integration. |
| **requirements.txt** | Lists the Python packages needed to run the app on Streamlit Community Cloud. |
| **README.md** | The file you’re reading now. It explains how to set up and use the dashboard. |

## What the app does

* Forecasts your daily cash balance from today until a future date (default: 2025‑12‑31) based on income, living costs, housing, phone, car insurance and storage bills.
* Lets you define **one‑time targets** (e.g. birthdays, holidays, attorney fees) and see how much you should set aside each week to fully fund them.
* Aggregates the daily model into a **weekly plan table** with income, fixed bills, daily living, sinking fund contributions, debt payment, weekly net, running balance and an **On‑Track?** flag.
* Displays a **line chart** of your projected balance and a **bar chart** comparing weekly income versus expenses.
* Shows upcoming bills and targets in the next 90 days and progress bars for each target.
* Optionally generates three weekly action suggestions if you provide an OpenAI API key.

The default values (income, expenses, targets) come from the problem description but you can change everything from the sidebar at any time.  All currency values are shown in USD and all dates use the format YYYY‑MM‑DD.

## Why this math is correct

* **Daily modelling.**  Each day you receive your daily income and pay your daily living cost.  Housing is charged once per week on Monday.  Phone, car insurance and storage bills occur on the 17th, 15th and 18th of each month respectively.  One‑time targets hit on their specific dates.  The daily net is `income − expenses` and the **balance** is the running sum of the net.
* **Sinking funds.**  For each target we compute the number of whole weeks left from today until its due date.  The **recommended weekly contribution** equals `target amount ÷ remaining weeks`.  Summing these contributions across all targets gives the “Target Set‑Aside” for each week.  These contributions do not change the actual daily cash flow; they are simply savings goals so you’re prepared when the due dates arrive.
* **Weekly plan.**  The app groups the daily schedule into calendar weeks (Monday–Sunday).  Income equals seven days of your daily income.  Fixed bills include housing and any monthly bills that fall within the week.  Daily living is seven days of your daily living cost.  The “Target Set‑Aside” column is the sum of all recommended weekly contributions.  **Net** = Income − (Fixed Bills + Daily Living + Target Set‑Aside + Debt Payment).  **Running Balance** is the cumulative sum of the weekly net.  A week is marked **On‑Track?** if your projected balance on or before each target’s due date remains non‑negative.
* **Example week.**  For the week starting **2025‑09‑22** (Monday): income = `$500 × 7 = $3,500`.  Fixed bills = `$665` housing (no monthly bills fall in that week).  Daily living = `$100 × 7 = $700`.  Summing the recommended sinking fund contributions for all six targets yields about `$1,567.68`.  Debt payment is not due that week.  Therefore `Net = $3,500 − ($665 + $700 + $1,567.68 + $0) ≈ $567.32`.  Running balance after this first week is `$567.32` and the **On‑Track?** flag is ✓ because there are no targets within that week and the projected balance before each target date stays positive.

## Setup (Non‑Coder)

Follow these steps to deploy and use your personal finance dashboard:

▢ **1. Create a GitHub repository.** Sign in to GitHub (free account) and make a new repository (e.g. `finance-dashboard`). It can be public or private.

▢ **2. Add the files.** Upload `streamlit_app.py`, `requirements.txt` and `README.md` to the root of the repository.  You can copy‑and‑paste their contents directly from the code blocks in the conversation if needed.

▢ **3. Commit and push.** Save (commit) the files and push them to GitHub.  This makes them available for deployment.

▢ **4. Deploy on Streamlit Community Cloud.** Go to [share.streamlit.io](https://share.streamlit.io) and sign in with your GitHub account.  Click **“New app”**, select your repository and choose `streamlit_app.py` as the entrypoint.

▢ **5. Wait for the build.** Streamlit will install the packages from `requirements.txt` and run the app.  Once complete you’ll see a URL like `https://your‑name.streamlit.app`.

▢ **6. Open your app.** Visit the URL.  The dashboard loads with your default values already filled in.  Explore the KPI cards, charts and weekly plan.

▢ **7. Adjust numbers.** Use the sidebar to change income, expenses, target dates/amounts and debt payment date.  The charts and tables update in real time.  Scroll down to see the calendar and target progress bars.

▢ **8. Save or load settings.** Click **“Download settings (JSON)”** to save your current configuration.  To restore it later, use **“Upload settings (JSON)”** in the sidebar and select the file you downloaded.

▢ **9. Enable AI advice (optional).** If you have an OpenAI API key, paste it into the **“OpenAI API key”** field in the sidebar and click **“Generate weekly advice”**.  Three concrete action suggestions will appear.  Without a key this section does nothing.

▢ **10. Troubleshoot.** If the app doesn’t build, check that all file names match exactly and that `requirements.txt` lists the correct packages.  If you see blank charts, ensure your end date is after today.  If you upload settings and the sidebar doesn’t update, refresh the page.

That’s it!  You can now monitor your cash flow, plan for upcoming expenses and stay on track.
