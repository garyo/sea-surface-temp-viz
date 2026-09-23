/**
 * Starts the nightly make-images workflow on time.
 *
 * GitHub's own `schedule:` trigger is best-effort and has been starting runs
 * hours late, while a workflow_dispatch starts within seconds.
 */

interface Env {
  GITHUB_TOKEN: string;
  GITHUB_REPO: string;
  WORKFLOW_FILE: string;
}

export default {
  async scheduled(_controller, env) {
    const url = `https://api.github.com/repos/${env.GITHUB_REPO}/actions/workflows/${env.WORKFLOW_FILE}/dispatches`;
    const res = await fetch(url, {
      method: "POST",
      headers: {
        Accept: "application/vnd.github+json",
        Authorization: `Bearer ${env.GITHUB_TOKEN}`,
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "sst-viz-nightly-trigger",
      },
      body: JSON.stringify({ ref: "main" }),
    });
    // Throwing marks the cron invocation as failed in the Cloudflare dashboard.
    if (!res.ok) {
      throw new Error(`Workflow dispatch failed: ${res.status} ${await res.text()}`);
    }
  },
} satisfies ExportedHandler<Env>;
