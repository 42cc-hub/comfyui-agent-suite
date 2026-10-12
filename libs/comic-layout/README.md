# comic-layout

把 N 张图排成**一页漫画**，并处理嵌字（对话框/旁白）。

- `scripts/comic_layout.py` —— 分格版式（多种模板、出血、间隔、页码）。
- `scripts/letter_page.py` —— 嵌字：文本框、描边、竖排，输出可直接投稿的页面。
- 参考：`references/comic_page_layout.md`。

依赖：`Pillow`（字体请自备，脚本用 `--font` 指定）。
