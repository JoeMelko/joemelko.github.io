# Personal Website

This repository contains a simple personal website. Open `index.html` in a web browser to view the main page.

The site includes:

- **About** section
- **Previous Work** section with sample project links
- **Blog** page that renders posts written in Markdown

Blog posts live in the `posts/` directory. Open a post by visiting
`blog.html?post=<slug>`, for example `blog.html?post=hello-world`.

No build step is required. Simply open the files directly in your browser.

If your browser shows "Error loading post." when loading a blog post directly from the file system, start a local server:

    python3 -m http.server

Then visit http://localhost:8000/blog.html?post=hello-world (or any slug). This avoids cross-origin restrictions that can block `fetch` from `file:` URLs.


