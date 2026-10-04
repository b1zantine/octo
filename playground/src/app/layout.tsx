// Modified for Octo branding and its local checkpoint API.
import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Octo playground",
  description: "Octo decision model: shared state, isolated questions, direct probability readout",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="en"
      className="h-full antialiased"
    >
      <body className="min-h-full flex flex-col">{children}</body>
    </html>
  );
}
