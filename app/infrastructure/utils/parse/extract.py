import re
import typing as tp
from dataclasses import dataclass

from bs4 import NavigableString, PageElement, Tag

from app.infrastructure.utils.parse.markdown import md
from app.infrastructure.utils.universal import normalize_string


def collapse_spaces(text: str) -> str:
    """Схлопывание пробелов (кроме лидирующих)"""
    lines = text.split("\n")

    processed_lines = []
    for line in lines:
        leading_spaces = re.match(r"(\s*)", line).group(1)
        rest_of_line = line[len(leading_spaces) :]
        processed_lines.append(leading_spaces + normalize_string(rest_of_line))

    text = "\n".join(processed_lines)
    return text


def extract_markdown_text(tag: Tag, **options: dict[str, tp.Any]) -> str:
    """Извлечение текста в формате Markdown"""
    return (
        collapse_spaces(md(html=tag.decode_contents(), **options))
        .replace("\n-\nGetting issue details... STATUS", "")
        .strip()
    )


def extract_clean_text(tag: Tag) -> str:
    """Извлечение текста в чистом формате"""
    # собираем весь текст в result
    result = []
    _process_tag(tag, result, list_stack=[])
    text = "".join(result)

    # очищаем множественные переносы строк
    text = re.sub(r"\n{3,}", "\n\n", text)

    lines = text.split("\n")

    # нормализуем строки, не теряя пробелы в их начале
    processed_lines = []
    for line in lines:
        leading_spaces = re.match(r"( *)", line).group(1)
        rest_of_line = line[len(leading_spaces) :]
        processed_lines.append(leading_spaces + normalize_string(rest_of_line))

    text = "\n".join(processed_lines)

    return text.strip()


def _process_tag(
    tag: Tag | NavigableString,
    result: list,
    list_stack: list,
    amount_of_indents: int = 0,
) -> None:
    """Рекурсивная обработка тега"""
    # если получили строку (NavigableString), добавляем в результат с учётом сдвига по пробелам
    if isinstance(tag, NavigableString):
        text = str(tag)
        if text:
            result.append(f"{amount_of_indents*' '}{text}")
        return

    # в противном случае работаем с тегом
    if isinstance(tag, Tag):
        # нумерованный список: добавляем переносы строк, нумерацию и пробелы
        # все вложенные теги / NavigableString рекурсивно обрабатываем, насыщяя result
        if tag.name == "ol":
            items = tag.find_all("li", recursive=False)

            for idx, item in enumerate(items):
                new_stack = list_stack + [idx + 1]

                prefix = ".".join(map(str, new_stack)) + ". "
                indent = " " * (len(new_stack) - 1)

                result.append("\n" + indent + prefix)
                _process_tag(item, result, new_stack, amount_of_indents=len(indent))
            return

        # маркированный список: добавляем переносы строк, дефис и пробелы
        # все вложенные теги / NavigableString рекурсивно обрабатываем, насыщяя result
        elif tag.name == "ul":
            items = tag.find_all("li", recursive=False)

            for _, item in enumerate(items):
                indent = " " * len(list_stack)
                result.append("\n" + indent + "- ")
                _process_tag(item, result, list_stack, amount_of_indents=len(indent))
            return

        # элемент списка: все вложенные теги / NavigableString рекурсивно обрабатываем, насыщяя result
        # увеличиваем list_stack для корректной работы с пробелами в маркированном списке
        elif tag.name == "li":
            child: PageElement
            for child in tag.children:
                _process_tag(
                    child,
                    result,
                    list_stack + [None] if child.name == "ul" else list_stack,
                    amount_of_indents=amount_of_indents,
                )
            return

        # пропускаем изображения
        elif tag.name in ["img", "image"]:
            return

        # учитываем переносы строк
        elif tag.name == "p":
            result.append("\n\n")
        elif tag.name == "br":
            result.append("\n")

        # не добавляем перенос строки для определённой группы тегов
        elif tag.name in [
            "b",
            "strong",
            "i",
            "em",
            "u",
            "s",
            "del",
            "mark",
            "sup",
            "sub",
            "small",
            "code",
            "kbd",
            "span",
            "a",
        ]:
            pass

        # для всех остальных тегов добавляем перенос строки
        else:
            result.append("\n")

        # рекурсивно обрабатываем все вложенные теги
        for child in tag.children:
            _process_tag(child, result, list_stack, amount_of_indents=amount_of_indents)


@dataclass
class CombinedRowsData:
    """Информация о строке с объединёнными столбцами"""

    rows: list[Tag]
    column_names: list[str]


def get_combined_rows(table_tag: Tag) -> list[CombinedRowsData]:
    """Возврат всех объединённых строк"""
    combined_rows_data: list[CombinedRowsData] = []
    all_rows = table_tag.find_all("tr")
    column_names = []

    for row_index, row in enumerate(all_rows):
        cells = row.find_all(["td", "th"])
        columns_with_merge = []
        for cell in cells:
            rowspan = cell.get("rowspan")
            if rowspan:
                try:
                    rowspan_value = int(rowspan)
                    if rowspan_value > 1:
                        header = find_header_for_cell(cell)
                        if not header:
                            raise Exception("Не удалось найти заголовок для ячейки")
                        combined_rows_data.append(
                            CombinedRowsData(
                                rows=[row],
                                column_names=[header.get_text(strip=True)],
                            )
                        )
                        for i in range(1, min(rowspan_value, len(all_rows) - row_index)):
                            combined_rows_data[-1].rows.append(all_rows[row_index + i])
                except (ValueError, TypeError):
                    continue
        if columns_with_merge:
            column_names += [columns_with_merge]

    grouped_data: dict[tuple[str], CombinedRowsData] = {}
    for item in combined_rows_data:
        rows_key = tuple(sorted([r.get_text(strip=True) for r in item.rows]))

        if rows_key in grouped_data:
            grouped_data[rows_key].column_names.extend(item.column_names)
        else:
            grouped_data[rows_key] = CombinedRowsData(
                rows=item.rows.copy(), column_names=item.column_names.copy()
            )

    result: list[CombinedRowsData] = []
    for item in grouped_data.values():
        unique_column_names = []
        for name in item.column_names:
            if name not in unique_column_names:
                unique_column_names.append(name)
        item.column_names = unique_column_names
        result.append(item)

    return result


def find_header_for_cell(cell_tag: Tag) -> Tag | None:
    """Поиск названия колонки для ячейки"""
    table = cell_tag.find_parent("table")
    if not table:
        return None

    cell_row = cell_tag.find_parent("tr")
    if not cell_row:
        return None

    all_rows = table.find_all("tr")

    try:
        cell_row_index = all_rows.index(cell_row)
    except ValueError:
        return None

    cell_siblings = cell_row.find_all(["td", "th"], recursive=False)
    try:
        cell_position = cell_siblings.index(cell_tag)
    except ValueError:
        return None

    if cell_row_index > 0:
        header_row = all_rows[0]
        header_cells = header_row.find_all(["th", "td"])

        if cell_position < len(header_cells):
            return header_cells[cell_position]

    return None
