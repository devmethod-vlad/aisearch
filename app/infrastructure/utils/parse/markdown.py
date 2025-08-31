import re
import typing as tp
from urllib.parse import urljoin

from bs4 import Tag
from markdownify import (
    ASTERISK,
    SPACES,
    STRIP,
    UNDERLINED,
    MarkdownConverter,
    chomp,
    re_line_with_content,
)

from app.infrastructure.utils.universal import is_absolute_url


class CustomMarkdownConverter(MarkdownConverter):
    """Кастомное расширение markdownify"""

    class DefaultOptions:
        autolinks = True
        bullets = "*+-"  # An iterable of bullet types.
        code_language = ""
        code_language_callback = None
        convert = None
        default_title = False
        escape_asterisks = True
        escape_underscores = True
        escape_misc = False
        heading_style = UNDERLINED
        keep_inline_images_in = []
        newline_style = SPACES
        strip = None
        strip_document = STRIP
        strong_em_symbol = ASTERISK
        sub_symbol = ""
        sup_symbol = ""
        table_infer_header = False
        wrap = False
        wrap_width = 80

    class Options(DefaultOptions):
        # custom realization START
        confluence_url = ""
        # custom realization END

    def convert_a(self, el: Tag, text: str, parent_tags: set[str]) -> str:
        """Конвертация ссылки в строку"""
        if "_noformat" in parent_tags:
            return text
        prefix, suffix, text = chomp(text)
        href = el.get("href")
        title = el.get("title")

        # custom realization START

        if not text or "/thumbnail/" in text:
            text = "Вложение"

        if not is_absolute_url(url=href):
            # and not href.startswith(self.options["confluence_url"]):
            # print(self.options["confluence_url"] == "")
            href = urljoin(self.options["confluence_url"], href)

        # custom realization END

        # For the replacement see #29: text nodes underscores are escaped
        if (
            self.options["autolinks"]
            and text.replace(r"\_", "_") == href
            and not title
            and not self.options["default_title"]
        ):
            # Shortcut syntax
            return "<%s>" % href  # noqa: UP031
        if self.options["default_title"] and not title:
            title = href
        title_part = ' "%s"' % title.replace('"', r"\"") if title else ""  # noqa: UP031

        return (
            "%s[%s](%s%s)%s" % (prefix, text, href, title_part, suffix)  # noqa: UP031
            if href
            else text
        )

    def convert_li(self, el: Tag, text: str, parent_tags: set[str]) -> str:
        """Конвертация элемента списка в строку"""
        # handle some early-exit scenarios
        text = (text or "").strip()
        if not text:
            return "\n"

        # custom realization START

        # determine list item bullet character to use
        parent = el.parent
        if parent is not None and parent.name == "ol":
            # Get all ancestor ol elements
            ol_ancestors = []
            current = el.parent
            while current is not None:
                if current.name == "ol":
                    ol_ancestors.insert(0, current)  # add to beginning to maintain order
                current = current.parent

            # Calculate the full number for nested lists
            full_number = []
            for ancestor_ol in ol_ancestors:
                if ancestor_ol.get("start") and str(ancestor_ol.get("start")).isnumeric():
                    start = int(ancestor_ol.get("start"))
                else:
                    start = 1
                # Find all li siblings before our element in this ol
                if ancestor_ol == el.parent:
                    siblings = el.find_previous_siblings("li")
                    position = start + len(siblings)
                else:
                    # For ancestor ols, find the li that contains the current ol
                    containing_li = None
                    for li in ancestor_ol.find_all("li"):
                        if li.find("ol") and (el in li.find_all() or li.find("ol") == el.parent):
                            containing_li = li
                            break
                    if containing_li:
                        siblings = containing_li.find_previous_siblings("li")
                        position = start + len(siblings)
                    else:
                        position = start
                full_number.append(str(position))

            bullet = ".".join(full_number) + "."
        else:
            # Handle unordered lists as before
            depth = -1
            current_el = el
            while current_el:
                if current_el.name == "ul":
                    depth += 1
                current_el = current_el.parent  # type: ignore
            bullets = self.options["bullets"]
            bullet = bullets[depth % len(bullets)]
        bullet = bullet + " "
        bullet_width = len(bullet)
        bullet_indent = " " * bullet_width

        # custom realization END

        # indent content lines by bullet width
        def _indent_for_li(match: re.Match) -> str:
            line_content = match.group(1)
            return bullet_indent + line_content if line_content else ""

        text = re_line_with_content.sub(_indent_for_li, text)

        # insert bullet into first-line indent whitespace
        text = bullet + text[bullet_width:]

        return "%s\n" % text  # noqa: UP031

    def convert_img(self, el: Tag, text: str, parent_tags: set[str]) -> str:
        """Конвертация img в текст"""
        alt = el.attrs.get("alt", None) or ""
        src = el.attrs.get("src", None) or ""
        title = el.attrs.get("title", None) or ""
        title_part = ' "%s"' % title.replace('"', r"\"") if title else ""  # noqa: UP031
        if "_inline" in parent_tags and el.parent.name not in self.options["keep_inline_images_in"]:
            return alt

        # custom realization START

        data_image_src = el.attrs.get("data-image-src", None) or ""

        # if el.get("class") and "unknown-attachment" in el["class"]:
        #     return ""

        if not is_absolute_url(url=src):
            # and not src.startswith(self.options["confluence_url"]):
            src = urljoin(self.options["confluence_url"], src)

        if "/thumbnail/" in src:
            return ""

        src = urljoin(self.options["confluence_url"], data_image_src or src)

        alt = "Вложение"

        return f"[{alt}]({src}{title_part})"

        # custom realization END


def md(html: str, **options: dict[str, tp.Any]) -> MarkdownConverter:
    """Инициализация Markdown-конвертера"""
    return CustomMarkdownConverter(**options).convert(html)
