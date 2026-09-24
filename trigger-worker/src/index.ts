/**
 * Starts the nightly make-images workflow as soon as NOAA's daily OISST update
 * is out.
 *
 * NOAA writes the previous day's preliminary OISST file at 13:31 UTC (its
 * Last-Modified, to the second, every day in Sept 2026), but it reaches the
 * public server some time later: at 14:16 UTC on 2026-09-24. That lag, plus a
 * possible hour's shift outside US daylight time, makes any fixed dispatch
 * time either early (missing the newest day) or late, so the cron ticks every
 * 10 minutes through a window and dispatches on the first tick that finds the
 * file. OISST is only one of the pipeline's sources, so from DEADLINE on a
 * tick dispatches regardless: a late or missing NOAA update must not cost the
 * day's run.
 *
 * Each tick first asks GitHub whether a run has started since the file was
 * written (or, without it, since the window opened). That keeps later ticks
 * from doubling up, retries a failed dispatch on the next tick, and counts a
 * manual run that already had the fresh data.
 *
 * Dispatching, rather than using the workflow's own `schedule:` trigger, is
 * because GitHub's cron is best-effort and was starting runs hours late; a
 * workflow_dispatch starts within seconds.
 */

interface Env {
  GITHUB_TOKEN: string;
  GITHUB_REPO: string;
  WORKFLOW_FILE: string;
}

const OISST_BASE =
  "https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation/v2.1/access/avhrr";
const USER_AGENT = "sst-viz-nightly-trigger";
const DAY_MS = 86_400_000;

// UTC minutes past midnight. WINDOW_OPEN must match the first hour of the cron
// in wrangler.jsonc, and DEADLINE must leave a few ticks before the window
// closes so a failed dispatch still gets retried.
const WINDOW_OPEN = 13 * 60;
const DEADLINE = 15 * 60 + 30;

function atMinuteOfDay(now: Date, minutes: number): Date {
  const midnight = Math.floor(now.getTime() / DAY_MS) * DAY_MS;
  return new Date(midnight + minutes * 60_000);
}

/** When NOAA posted the preliminary OISST file for the day before `now`, or null if it isn't up yet. */
async function oisstPostedAt(now: Date): Promise<Date | null> {
  const ymd = new Date(now.getTime() - DAY_MS).toISOString().slice(0, 10).replaceAll("-", "");
  const url = `${OISST_BASE}/${ymd.slice(0, 6)}/oisst-avhrr-v02r01.${ymd}_preliminary.nc`;
  try {
    const res = await fetch(url, { method: "HEAD", headers: { "User-Agent": USER_AGENT } });
    const lastModified = res.ok ? res.headers.get("last-modified") : null;
    return lastModified ? new Date(lastModified) : null;
  } catch {
    // Unreachable counts as not posted; the deadline still guarantees a run.
    return null;
  }
}

async function github(env: Env, path: string, init?: RequestInit): Promise<Response> {
  const url = `https://api.github.com/repos/${env.GITHUB_REPO}/actions/workflows/${env.WORKFLOW_FILE}/${path}`;
  const res = await fetch(url, {
    ...init,
    headers: {
      Accept: "application/vnd.github+json",
      Authorization: `Bearer ${env.GITHUB_TOKEN}`,
      "X-GitHub-Api-Version": "2022-11-28",
      "User-Agent": USER_AGENT,
    },
  });
  // Throwing marks the cron invocation as failed in the Cloudflare dashboard.
  if (!res.ok) {
    throw new Error(`GitHub ${init?.method ?? "GET"} ${path} failed: ${res.status} ${await res.text()}`);
  }
  return res;
}

async function hasRunSince(env: Env, since: Date): Promise<boolean> {
  const query = new URLSearchParams({ created: `>=${since.toISOString()}`, per_page: "1" });
  const res = await github(env, `runs?${query}`);
  const { total_count } = await res.json<{ total_count: number }>();
  return total_count > 0;
}

export default {
  async scheduled(controller, env) {
    const now = new Date(controller.scheduledTime);
    const posted = await oisstPostedAt(now);
    if (!posted && now < atMinuteOfDay(now, DEADLINE)) {
      console.log("OISST not posted yet; waiting");
      return;
    }
    const since = posted ?? atMinuteOfDay(now, WINDOW_OPEN);
    if (await hasRunSince(env, since)) {
      console.log(`A run already started since ${since.toISOString()}`);
      return;
    }
    await github(env, "dispatches", { method: "POST", body: JSON.stringify({ ref: "main" }) });
    console.log(posted ? `Dispatched; OISST posted ${posted.toISOString()}` : "Dispatched at deadline without new OISST");
  },
} satisfies ExportedHandler<Env>;
