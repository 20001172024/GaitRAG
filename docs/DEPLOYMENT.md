# Deploying GaitRAG (Hugging Face Spaces + Neon)

The app runs as one Docker container (API + dashboard) on a free Hugging Face Space,
with a free Neon Postgres database (pgvector) for the paper index. GitHub Actions does
the rest: every push to `main` that passes CI redeploys the Space.

Check current free-tier limits on both sites before relying on them; they change.

## One-time setup (about 20 minutes)

### 1. Database: Neon

1. Sign up at <https://neon.tech> and create a project (any region close to you).
2. Copy the **connection string** from the dashboard. It looks like
   `postgresql://user:password@ep-xxx.region.aws.neon.tech/neondb?sslmode=require`.
3. Nothing else is needed: the app's migrations run `CREATE EXTENSION vector`.

### 2. App host: Hugging Face

1. Sign up at <https://huggingface.co>.
2. Create an **access token** with **Write** permission:
   Settings → Access Tokens → New token → type *Write*.
3. Choose a Space name, e.g. `your-hf-username/gaitrag`. You do not need to create it;
   the deploy workflow does.

### 3. GitHub repository settings

In the GitHub repo: **Settings → Secrets and variables → Actions**.

| Kind | Name | Value |
|---|---|---|
| Secret | `HF_TOKEN` | the Hugging Face write token |
| Secret | `DATABASE_URL` | the Neon connection string |
| Secret | `OPENAI_API_KEY` | your OpenAI key (optional: without it the app works but gives no explanations) |
| Variable | `HF_SPACE` | `your-hf-username/gaitrag` |
| Variable (optional) | `LLM_MODEL` | e.g. `gpt-4o-mini` (the default) |

Secrets stay in GitHub and are copied into the Space's own secret settings by the
deploy script. They are never written to the repository.

### 4. Merge and deploy

1. Merge the pull request into `main`.
2. CI runs; when it passes, **Deploy** runs and uploads the app to the Space.
   (You can also start it by hand: Actions → Deploy → Run workflow.)
3. Hugging Face builds the image (about 10 minutes the first time). Follow it on the
   Space page under *Logs*.

### 5. Build the paper index (once)

Actions → **Ingest papers** → Run workflow. It downloads the 24 open-access papers
and indexes them into Neon (about 10–15 minutes on a GitHub runner).

### 6. Check it

Open `https://huggingface.co/spaces/your-hf-username/gaitrag` and click **Normal walk**.
`/health` should report `"db":"ok"`, and `/papers` should list 24 papers.

## Later

- **After approving norms** in `data/papers/norms_review.csv`: commit, push to `main`,
  then run **Ingest papers** again. It only loads the approved rows; unchanged papers are
  skipped.
- **Cost control:** `/ask` is limited to 20 questions per visitor per hour and 300 per
  day overall (variables `ASK_LIMIT_PER_CLIENT_PER_HOUR`, `ASK_LIMIT_PER_DAY`). Also set
  a monthly spending limit in your OpenAI account.
- **Sleeping:** free Spaces sleep after a period without visitors; the first visit
  after that takes a minute or two while the container starts.
- **Rotating secrets:** update the GitHub secret, then re-run **Deploy**.

## Run the production image elsewhere

The same image runs anywhere Docker runs:

```bash
docker build -t gaitrag .
docker run -p 8000:8000 -e DATABASE_URL=... -e OPENAI_API_KEY=... gaitrag          # serve
docker run --rm -e DATABASE_URL=... gaitrag ingest                                  # index papers
```

It needs about 1–1.5 GB of RAM for the models.
