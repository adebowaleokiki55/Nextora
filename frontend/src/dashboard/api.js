export const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

async function handle(response) {
  const text = await response.text();
  const data = text ? JSON.parse(text) : null;
  if (!response.ok) {
    const detail = Array.isArray(data?.detail)
      ? data.detail.map((item) => item.msg).join(" ")
      : data?.detail;
    throw new Error(detail || `Request failed (${response.status}).`);
  }
  return data;
}

function friendly(error) {
  if (error.message === "Failed to fetch") {
    return new Error("Cannot reach the Nextora backend. Is it running?");
  }
  return error;
}

export async function apiGet(path) {
  try {
    return await handle(await fetch(`${API_URL}${path}`));
  } catch (error) {
    throw friendly(error);
  }
}

export async function apiPost(path, body) {
  try {
    return await handle(
      await fetch(`${API_URL}${path}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: body === undefined ? undefined : JSON.stringify(body),
      })
    );
  } catch (error) {
    throw friendly(error);
  }
}

export function eventsUrl(runId) {
  return `${API_URL}/api/command/runs/${runId}/events`;
}
