import { LoginForm } from "@/components/login-form";

type LoginPageProps = {
  searchParams?: Promise<{ error?: string }>;
};

export default async function LoginPage({ searchParams }: LoginPageProps) {
  const params = await searchParams;

  return (
    <div className="auth-page">
      <div className="text-center">
        <h1 className="display text-2xl md:text-3xl">Account</h1>
        <p className="mt-1.5 text-sm text-[var(--fg-muted)]">
          Sign in to save your voice and conversations.
        </p>
      </div>
      <LoginForm initialError={params?.error} />
    </div>
  );
}
