import { LoginForm } from "@/components/login-form";

export default function LoginPage() {
  return (
    <div className="space-y-6">
      <div className="text-center">
        <h1 className="display text-2xl md:text-3xl">Account</h1>
        <p className="mt-1.5 text-sm text-[var(--fg-muted)]">
          Sign in to save your voice and conversations.
        </p>
      </div>
      <LoginForm />
    </div>
  );
}
