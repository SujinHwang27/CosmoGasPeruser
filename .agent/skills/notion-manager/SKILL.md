---
name: notion-manager
description: Uses Notion MCP to execute tasks in Notion
---

### Database
Default database is "Antigravity Log".

### Insert .md file as a row in database
1. Title is the filename without extension.
2. Parse all the content of the file into blocks in Notion including tables and mathematical equations. Nothing should be missing. 
3. Inspect if there is an existing entry with the same title. If so, only update the date and content. Else, create a new page in the database with the title and content.
