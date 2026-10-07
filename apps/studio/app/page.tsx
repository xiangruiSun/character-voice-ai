"use client";

import { useRouter } from "next/navigation";
import { useEffect } from "react";

// The product starts in Chat, which also guides first-time users through setup.
export default function Home() {
  const router = useRouter();
  useEffect(() => { router.replace("/chat"); }, [router]);
  return null;
}
