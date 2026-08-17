# sush4nt.github.io

Personal site and blog, built with [Hugo](https://gohugo.io/) using the [PaperMod](https://github.com/adityatelange/hugo-PaperMod) theme, hosted on GitHub Pages.

Live at: https://sush4nt.github.io/

**This repo is public and should only ever contain content meant for a public audience.** Purely personal/private notes live in a separate private repo (`sush4nt/personal-docs`), deployed to a password-gated Cloudflare Pages site instead — see that repo's README for details. Do not use `draft: true` as a way to "hide" sensitive content here; drafts are excluded from the rendered site but the raw markdown is still visible to anyone browsing this public repo on GitHub.

## How it's built & served

- **Hugo** turns the Markdown files in `content/` into static HTML using the `PaperMod` theme (included as a git submodule in `themes/PaperMod`).
- Config lives in `hugo.yaml` (site title, menu, social links, favicon, etc.).
- On every push to `main`, the GitHub Actions workflow in `.github/workflows/hugo.yaml` builds the site with Hugo and deploys the output to **GitHub Pages** — no manual build/deploy step needed.

## Running locally

```bash
# one-time setup
brew install hugo
git clone --recurse-submodules https://github.com/sush4nt/sush4nt.github.io.git
cd sush4nt.github.io

# preview the site (matches what's live)
hugo server
```

Open http://localhost:1313/. The server live-reloads as you edit files.

## Pushing content

```bash
# create a new post
hugo new content posts/my-new-post.md

# edit content/posts/my-new-post.md, then set draft: false when ready to publish

git add -A
git commit -m "Add my-new-post"
git push
```

Pushing to `main` triggers the Actions workflow, and the change is live within ~30–60 seconds.

## Additional info

- **Front matter** on each post should include `title`, `date`, `tags`, and a short `summary` (shown in the post list).
- **Theme submodule**: since PaperMod is a git submodule, always clone with `--recurse-submodules`, or run `git submodule update --init --recursive` after a normal clone.
- **Favicon**: `static/favicon.svg` — an SVG containing the 👋 emoji, rendered via the browser's own emoji font.
- **Pages settings**: on GitHub, Pages is configured to deploy from GitHub Actions (Settings → Pages → Source: GitHub Actions).
