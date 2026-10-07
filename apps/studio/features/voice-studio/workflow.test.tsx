import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { Stepper } from "@/components/shared/blocks";
import { TrainingStatusBadge } from "@/components/shared/status";
import { CharacterWizard } from "@/features/characters/character-wizard";
import type { TrainingJob, VoiceModel, VoicePack } from "@/lib/types";

import { stepFor } from "./workflow";

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn(), back: vi.fn(), prefetch: vi.fn() }),
  useSearchParams: () => new URLSearchParams(),
  usePathname: () => "/",
}));

afterEach(cleanup);

const pack = (status: VoicePack["status"]) => ({ status } as VoicePack);
const job = (status: TrainingJob["status"]) => ({ status } as TrainingJob);
const model = (approved: boolean) => ({ approved } as VoiceModel);

describe("voice studio resumes where the server says", () => {
  it.each([
    ["new pack", pack("empty"), undefined, undefined, 0],
    ["processing", pack("processing"), undefined, undefined, 1],
    ["needs review", pack("needs_review"), undefined, undefined, 1],
    ["ready to train", pack("ready"), undefined, undefined, 3],
    ["training running", pack("ready"), job("training"), undefined, 3],
    ["training failed", pack("ready"), job("failed"), undefined, 3],
    ["trained, untested", pack("ready"), job("completed"), model(false), 4],
    ["approved voice", pack("ready"), job("completed"), model(true), 5],
  ])("%s", (_name, p, j, m, expected) => {
    expect(stepFor(p, j, m)).toBe(expected);
  });
});

describe("training status display", () => {
  it.each([
    ["queued", "排队中"], ["training", "进行中"], ["completed", "已完成"],
    ["failed", "失败"], ["cancelled", "已取消"], ["cancel_requested", "正在取消"],
  ])("%s → icon + text", (status, label) => {
    const { container } = render(<TrainingStatusBadge status={status} />);
    expect(screen.getByText(label)).toBeTruthy();
    expect(container.querySelector("svg")).toBeTruthy();          // never colour alone
  });
});

describe("stepper navigation", () => {
  it("lets you go back to finished steps but not skip ahead", () => {
    const onSelect = vi.fn();
    render(<Stepper current={2} onSelect={onSelect}
                    steps={[{ key: "a", label: "上传" }, { key: "b", label: "检查" }, { key: "c", label: "准备" },
                            { key: "d", label: "训练", disabled: true }]} />);
    fireEvent.click(screen.getByRole("button", { name: /上传/ }));
    expect(onSelect).toHaveBeenCalledWith(0);
    expect((screen.getByRole("button", { name: /训练/ }) as HTMLButtonElement).disabled).toBe(true);
    expect(screen.getByRole("button", { name: /准备/ }).getAttribute("aria-current")).toBe("step");
  });
});

describe("character wizard validation", () => {
  it("cannot advance without a name", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response("[]", { status: 200 })));
    render(
      <QueryClientProvider client={new QueryClient()}>
        <CharacterWizard open onOpenChange={() => {}} />
      </QueryClientProvider>,
    );
    const next = await screen.findByRole("button", { name: /下一步/ });
    expect((next as HTMLButtonElement).disabled).toBe(true);
    fireEvent.change(screen.getByLabelText("角色名称"), { target: { value: "小测" } });
    expect((screen.getByRole("button", { name: /下一步/ }) as HTMLButtonElement).disabled).toBe(false);
    vi.unstubAllGlobals();
  });
});
