

----
name: "better-auth-integrator"
description: "Expert skill that analyzes any existing web project (any framework or static site), identifies its architecture, then seamlessly implements full-featured authentication using Better-Auth with Google, Apple, Facebook, and X (Twitter) OAuth + email/password. Includes onboarding questionnaire for user background personalization, Better-Auth-UI components, header auth buttons, and dedicated Sign-in/Sign-up pages. Handles database setup intelligently (uses existing DB or creates new one with user confirmation). Creates separate backend if project is static or SSR-incompatible."
version: "1.2.0"
----

# Better-Auth Full Integration Skill

## When to Use This Skill

- User wants to add complete authentication (Sign-up + Sign-in) with social logins (Google, Apple, Facebook, X)
- User mentions Better-Auth or wants the most modern, secure, TypeScript-first auth solution
- Project may be Next.js, Remix, Astro, Vite, SvelteKit, Nuxt, pure React, or even a completely static site
- User wants personalized onboarding questions after signup (software/hardware background)
- User needs beautiful auth UI out of the box (Better-Auth-UI)
- User wants header buttons (Sign In / Sign Up / Sign Out) + dedicated auth pages

## How This Skill Works (Step-by-Step Execution)

1. **Project Analysis Phase**
   - Detect framework (Next.js App Router, Pages Router, Remix, Astro, SvelteKit, etc.)
   - Determine if SSR/SSG/static
   - Check for existing backend (Node/Express, tRPC, Hono, etc.)
   - Detect existing database setup and ORM (Prisma, Drizzle, TypeORM, raw SQL, etc.)

2. **Database Strategy**
   - If a database is already configured → ask for connection string/credentials (only after explicit user confirmation)
   - If no database → ask user preference (PostgreSQL recommended) and guide creation (e.g., Supabase, Neon, Railway, PlanetScale)
   - Only after user says “yes, go ahead”, create required Better-Auth tables/schema automatically

3. **Backend Setup**
   - If project supports SSR/API routes → add Better-Auth directly inside the project
   - If static or incompatible → create separate backend repo/folder using Hono + Node.js/Express (deployable to Vercel, Railway, Fly.io, etc.)
   - Always use Drizzle ORM (Better-Auth’s preferred adapter) with PostgreSQL

4. **Better-Auth Core Implementation**
   - Install `better-auth` + `better-auth-ui` + required plugins (google, apple, facebook, twitter)
   - Configure env variables for all OAuth providers
   - Set up email/password + magic links as fallback
   - Add user metadata table/extension for background questionnaire answers

5. **Onboarding Flow**
   - After first successful sign-up → redirect to `/onboarding`
   - Multi-step form asking:
     - Years of programming experience
     - Primary languages/frameworks
     - Hardware (Mac/Windows/Linux, CPU, GPU, RAM)
     - Development focus (frontend/backend/full-stack/AI/ML/mobile)
   - Store answers in `user_metadata` table for future personalization

6. **UI Integration (Better-Auth-UI)**
   - Create `/sign-in`, `/sign-up` pages with `<AuthUI />`
   - Add header component with dynamic auth buttons:
     - Guest → “Sign In” | “Sign Up”
     - Logged in → Avatar + Dropdown with “Profile”, “Settings”, “Sign Out”
   - Fully responsive, accessible, dark mode ready

7. **Session Management & Protection**
   - Server-side session validation
   - `authClient` React hook for frontend state
   - Protected routes middleware (Next.js middleware or route guards)

## Output You Will Receive

After activation, I will deliver:

- Complete step-by-step implementation plan tailored to your exact project
- Exact terminal commands to run
- File-by-file code changes/additions
- `.env.example` with all required variables
- Database schema (Drizzle migrations)
- Separate backend repo link (if needed)
- Ready-to-copy onboarding questionnaire component
- Header component with conditional auth UI
- Fully working, production-ready authentication system

## Example Usage

**User says:**  
“I have a Next.js 14 App Router site using Prisma + PostgreSQL on Supabase. Add Better-Auth with Google, Apple, Facebook, X login and ask users about their dev experience after signup.”

**This Skill Instantly Activates → Delivers:**

- Confirmed DB usage (no new DB needed)
- Prisma → Drizzle migration plan (or dual-ORM strategy if preferred)
- `/src/auth` folder structure with full Better-Auth config
- All OAuth callbacks configured
- `/app/(auth)/sign-in/[[...better-auth]]/page.tsx` using `<AuthUI />`
- Header with dynamic auth state
- `/app/onboarding/page.tsx` with questionnaire
- Protected route example using middleware

**User says:**  
“My site is a static Astro + React site. Just add login with social providers.”

**This Skill Responds:**  
→ Creates separate `better-auth-backend/` folder (Hono + Node)  
→ Deploys in < 2 minutes to Vercel  
→ Adds minimal client `authClient` to Astro  
→ Injects header buttons + modal sign-in using Better-Auth-UI  
→ Full social login working on a 100% static frontend

## Activate This Skill By Saying

- “Add Better-Auth to my project”
- “Implement signup and login with Google/Apple/Facebook/X”
- “I want Better-Auth with onboarding questions”
- “Set up authentication for my [Next.js/Astro/Remix/etc.] site”