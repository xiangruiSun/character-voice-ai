// The only module that knows how to talk to the backend. Same-origin by design: the
// browser never learns where a model runs.

export class ApiError extends Error {
  status: number;
  hint: string | null;

  constructor(status: number, message: string, hint: string | null = null) {
    super(message);
    this.status = status;
    this.hint = hint;
  }
}

async function parseError(response: Response): Promise<ApiError> {
  let message = `请求失败（${response.status}）`;
  let hint: string | null = null;
  try {
    const body = await response.json();
    const detail = body?.detail;
    if (typeof detail === "string") message = detail;
    else if (detail?.message) {
      message = detail.message;
      hint = detail.hint ?? null;
    } else if (Array.isArray(detail) && detail[0]?.msg) {
      message = "填写的内容不完整或格式不正确";
      hint = detail.map((d: { loc?: string[]; msg: string }) => `${(d.loc ?? []).slice(-1)[0] ?? ""}: ${d.msg}`).join("；");
    }
  } catch {
    /* not JSON */
  }
  if (response.status === 502 || response.status === 504) {
    message = "无法连接到本地后端";
    hint = "请确认后端服务正在运行";
  }
  return new ApiError(response.status, message, hint);
}

async function request<T>(method: string, path: string, body?: unknown): Promise<T> {
  let response: Response;
  try {
    response = await fetch(path, {
      method,
      headers: body === undefined ? undefined : { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  } catch {
    throw new ApiError(0, "无法连接到本地后端", "请确认后端服务正在运行");
  }
  if (!response.ok) throw await parseError(response);
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

export const api = {
  get: <T>(path: string) => request<T>("GET", path),
  post: <T>(path: string, body?: unknown) => request<T>("POST", path, body ?? {}),
  patch: <T>(path: string, body: unknown) => request<T>("PATCH", path, body),
  delete: (path: string) => request<void>("DELETE", path),
};

/** Upload with real progress (fetch cannot report upload progress). */
export function uploadFiles<T>(path: string, files: File[], field: string,
                               onProgress: (fraction: number) => void): Promise<T> {
  return new Promise((resolve, reject) => {
    const form = new FormData();
    files.forEach((file) => form.append(field, file, file.name));
    const xhr = new XMLHttpRequest();
    xhr.open("POST", path);
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable) onProgress(event.loaded / event.total);
    };
    xhr.onload = async () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(JSON.parse(xhr.responseText) as T);
      } else {
        reject(await parseError(new Response(xhr.responseText, { status: xhr.status })));
      }
    };
    xhr.onerror = () => reject(new ApiError(0, "上传中断", "请检查后端是否仍在运行后重试"));
    xhr.send(form);
  });
}

/** WebSocket URL for a backend path. In `next dev` (:3000) the socket goes to :8000. */
export function wsUrl(path: string): string {
  const { protocol, hostname, port, host } = window.location;
  const scheme = protocol === "https:" ? "wss:" : "ws:";
  const target = port === "3000" ? `${hostname}:8000` : host;
  return `${scheme}//${target}${path}`;
}

export function errorText(error: unknown): { message: string; hint: string | null } {
  if (error instanceof ApiError) return { message: error.message, hint: error.hint };
  if (error instanceof Error) return { message: error.message, hint: null };
  return { message: "出了点问题", hint: null };
}
