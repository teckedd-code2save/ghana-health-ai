import { LoginForm } from "@/components/login-form";

export default function LoginPage() {
  return (
    <div className="space-y-4">
      <div className="text-center">
        <h1 className="font-[family-name:var(--font-display)] text-3xl">Account</h1>
        <p className="text-sm text-[var(--fg-muted)]">Session cookies · consent for voice & health</p>
      </div>
      <LoginForm />
    </div>
  );
}
