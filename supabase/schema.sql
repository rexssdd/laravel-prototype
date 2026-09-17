create table if not exists public.analysis_results (
    id uuid primary key,
    filename text not null,
    result jsonb not null,
    created_at timestamptz not null default now()
);

alter table public.analysis_results enable row level security;

revoke all on public.analysis_results from anon, authenticated;