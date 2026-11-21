# Nextora demo script (about 6 minutes)

A step-by-step walkthrough of the Ikorodu scenario for judges. **Bold** is what to click; *italics* is what to say.

## Before you present (10 minutes earlier)

1. Start the backend: `python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000`
2. Start the frontend: `cd frontend` then `npm run dev`
3. Open http://localhost:5173/?view=command and press **Seed demo data** (about 20 seconds). Check that:
   - *Highest risk* shows **Ikorodu, Very high**;
   - *Supply shortages* shows **3**;
   - *Awaiting approval* shows **0**.
4. Type your name into the approval queue's **Your name** box (it is remembered).
5. Decide the agent mode:
   - **Live AI** if `GET /api/health` shows `agent_configured: true` *and* a quick test run succeeded (an invalid key makes live runs fail);
   - otherwise **Offline demo (scripted)**. It needs no key or internet and is labelled as scripted on screen.
6. Have a blood-smear image ready for the lab step (a thin-smear photo you have permission to use). Upload it as a demo image (leave "real patient sample" unticked).
7. If the venue's internet is unreliable, map tiles may not load; markers, charts and everything else still work.

## 1. The problem (30 seconds)

*Malaria teams in Lagos get signals from different places: weather, clinic reports, lab results and stock levels. Nextora joins them in one command centre so a team can see risk rising and act before it gets worse — with a human approving every action.*

Point at the amber banner: *Case counts, facilities and stock in this demo are synthetic. Weather is real reanalysis data. We label synthetic data everywhere you see it.*

## 2. Surveillance and prediction (1.5 minutes)

1. Point at the **map**. *Each circle is a local government area, coloured by forecast risk. Ikorodu is red.*
2. **Click Ikorodu.** In the forecast chart:
   - *The black line is reported cases: far above the usual level for this time of year.*
   - *The orange dashed line is our 1–4 week forecast, with its 80% range. The red dotted line is Ikorodu's alert threshold: the classic epidemic rule, mean plus two standard deviations of earlier years.*
   - *The risk score of 90 is the highest estimated chance, over the next four weeks, that weekly cases reach that threshold: about 90% for next week.*
3. In **Surveillance signals**: *Cases and the 4-week trend are high, but real rainfall is normal. So weather doesn't explain this rise — that's a reason to investigate, not to assume.*
4. Scroll to **Forecast accuracy vs baselines**: *We always compare against simple baselines. Our model beats all three on case numbers. For yes/no alerts, simply assuming next week equals this week does about as well — we report that honestly. And this is synthetic data, so it isn't proof of real-world accuracy.*

## 3. Supply (45 seconds)

1. Point at **Supply shortages: 3** and the red facility dots near Ikorodu.
2. In the location panel's **Facility stock** table: *Centre B has no test kits at all, A has under half a week left, C under two weeks. Without tests, they can't confirm cases during a surge.*
3. *All these numbers come from plain, tested code, not from the AI: weeks of stock, how many kits are needed, and which nearby facilities can spare stock without dropping below six weeks of their own supply.*

## 4. AI command centre (1.5 minutes)

1. In **AI command centre**, check the location is **Ikorodu** and pick the mode you chose. Press **Run response**.
2. As the feed streams, narrate:
   - *The Commander reviews the situation, then hands tasks to four specialists.*
   - *The Surveillance agent pulls the signals and forecast.*
   - *The Supply agent gets the redistribution plan and puts it in the approval queue. It can't invent quantities — they come from the plan.*
   - *The Outreach agent drafts one prevention SMS.*
   - *The Policy agent writes a situation report.*
3. Open one **tool result** to show the real data the agent saw.
4. Scroll to the **situation report**: *Note the data-limitations section — it says which data is synthetic.*

If you used offline mode: *This is our offline backup: a scripted sequence calling the same real tools, labelled as scripted.*

## 5. Human approval (1 minute)

1. In **Approval queue**: *Nothing happens until a named person approves.*
2. **Approve** one transfer to Centre B. *Approving actually moves the stock. The system re-checks the donor's stock at that moment and refuses if the plan is out of date.*
3. Press **Refresh**: Centre B's status improves, and once its shortage is covered its alert resolves.
4. **Reject** one item with a note, e.g. *"Use central stock instead"*: *Rejected proposals do nothing.*
5. Read the SMS draft aloud. *Approving it only queues it; no real SMS is sent in this demo.*

## 6. Laboratory intelligence (1 minute, optional)

1. Open **Cell Explorer** from the header. Choose **Whole Smear** mode and upload the smear image.
2. *The model was trained on single cells from the NIH dataset, so we first separate the cells with OpenCV, then classify each one: green uninfected, red possibly parasitized.*
3. Use **Submit for review** (laboratory: Ikorodu Demo Laboratory). Then open **Lab review**.
4. Verify it with your own reading. *Only a qualified professional's verified result feeds surveillance. This is decision support, not a diagnostic tool.*

## Closing (15 seconds)

*Nextora turns scattered signals into one picture and a coordinated response, with the maths in code, honest labels on synthetic data, and a human approving every consequential action.*

## If something goes wrong

| Problem | What to do |
|---|---|
| Live run fails (e.g. provider error) | Switch the mode to **Offline demo** and run again |
| Map is blank | Internet issue with map tiles; the rest of the dashboard still works |
| Numbers look different from this script | Press **Seed demo data** to reset the scenario |
| "Cannot reach the Nextora backend" | Start the backend; check `VITE_API_URL` |
| An approval says FAILED | Expected if stock changed since the plan: the safety check worked. Run the response again for a fresh plan |
