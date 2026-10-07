import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, renderHook } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { keys } from "@/lib/queries";
import type { TrainingJob } from "@/lib/types";

import { useJobStream } from "./training";

class FakeEventSource {
  static opened = 0;
  constructor(public url: string) { FakeEventSource.opened += 1; }
  addEventListener() {}
  close() {}
}

beforeEach(() => {
  FakeEventSource.opened = 0;
  vi.stubGlobal("EventSource", FakeEventSource);
});
afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

const job = (status: TrainingJob["status"]) => ({ id: "tj_1", status } as TrainingJob);

function setup() {
  const client = new QueryClient();
  const invalidate = vi.spyOn(client, "invalidateQueries");
  const wrapper = ({ children }: { children: ReactNode }) => <QueryClientProvider client={client}>{children}</QueryClientProvider>;
  const hook = renderHook(({ initial }) => useJobStream("tj_1", initial), { wrapper, initialProps: { initial: job("training") } });
  return { hook, invalidate };
}

describe("training job stream", () => {
  it("keeps one stream open while the job list refetches", () => {
    const { hook } = setup();
    hook.rerender({ initial: job("training") });   // a refetch: same job, new object
    hook.rerender({ initial: job("training") });
    expect(FakeEventSource.opened).toBe(1);
  });

  it("re-reads voice models when the list learns of completion before the stream does", () => {
    const { hook, invalidate } = setup();
    hook.rerender({ initial: job("completed") });
    const invalidated = invalidate.mock.calls.map(([filters]) => filters?.queryKey);
    expect(invalidated).toContainEqual(keys.voices);
    expect(invalidated).toContainEqual(keys.packs);
  });

  it("does not refetch for a job that was already finished when opened", () => {
    const client = new QueryClient();
    const invalidate = vi.spyOn(client, "invalidateQueries");
    const wrapper = ({ children }: { children: ReactNode }) => <QueryClientProvider client={client}>{children}</QueryClientProvider>;
    renderHook(() => useJobStream("tj_1", job("completed")), { wrapper });
    expect(FakeEventSource.opened).toBe(0);
    expect(invalidate).not.toHaveBeenCalled();
  });
});
