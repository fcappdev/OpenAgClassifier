# based on dicttoxml.py from https://github.com/quandyfactory/dicttoxml/blob/master/dicttoxml.py
from random import randint
import numbers
import collections
from xml.dom.minidom import parseString


def make_id(element, start=100000, end=999999):
    """
    Returns a random integer
    """

    return '%s_%s' % (element, randint(start, end))


def get_unique_id(element):
    """
    Returns a unique id for a given element
    """

    this_id = make_id(element)
    ids = []
    dup = True
    while dup:
        if this_id not in ids:
            dup = False
            ids.append(this_id)
        else:
            this_id = make_id(element)
    return ids[-1]


def get_xml_type(val):
    if isinstance(val, bool):
        return 'bool'
    if isinstance(val, str):
        return 'str'
    if isinstance(val, int):
        return 'int'
    if isinstance(val, float):
        return 'float'
    if val is None:
        return 'null'
    if isinstance(val, dict):
        return 'dict'
    if isinstance(val, collections.Iterable):
        return 'list'
    return type(val).__name__


def escape_xml(s):
    if isinstance(s, str):
        s = s.replace('&', '&amp;')
        s = s.replace('"', '&quot;')
        s = s.replace('\'', '&apos;')
        s = s.replace('<', '&lt;')
        s = s.replace('>', '&gt;')
    return s


def make_attrstring(attr):
    """
    Returns an attribute string in the form key='val'
    """

    attr_string = ' '.join(['%s="%s"' % (k, v) for k, v in attr.items()])
    return '%s%s' % (' ' if attr_string != '' else '', attr_string)


def key_is_valid_xml(key):
    """
    Checks that a key is a valid XML name
    """
    test_xml = '<?xml version="1.0" encoding="UTF-8" ?><%s>foo</%s>' % (key, key)
    try:
        parseString(test_xml)
        return True
    except Exception as ex:
        print(ex)
        return False


def make_valid_xml_name(key, attr):
    """
    Tests an XML name and fixes it if invalid
    """

    key = escape_xml(key)
    attr = escape_xml(attr)

    # pass through if key is already valid
    if key_is_valid_xml(key):
        return key, attr

    # prepend a lowercase n if the key is numeric
    if key.isdigit():
        return 'n%s' % key, attr

    # replace spaces with underscores if that fixes the problem
    if key_is_valid_xml(key.replace(' ', '_')):
        return key.replace(' ', '_'), attr

    # key is still invalid - move it into a name attribute
    attr['name'] = key
    key = 'key'
    return key, attr


def wrap_cdata(s):
    """Wraps a string into CDATA sections"""
    s = str(s).replace(']]>', ']]]]><![CDATA[>')
    return '<![CDATA[' + s + ']]>'


def default_item_func(_):
    return 'item'


def convert(obj, ids, attr_type, item_func, cdata, parent='root'):
    """
    Routes the elements of an object to the right function to convert them
    based on their data type
    """

    item_name = item_func(parent)

    if isinstance(obj, str):
        return convert_kv(item_name, obj, attr_type, cdata)
    if hasattr(obj, 'isoformat'):
        return convert_kv(item_name, obj.isoformat(), attr_type, cdata)
    if type(obj) == bool:
        return convert_bool(item_name, obj, attr_type, cdata)
    if obj is None:
        return convert_none(item_name, '', attr_type, cdata)
    if isinstance(obj, dict):
        return convert_dict(obj, ids, parent, attr_type, item_func, cdata)
    if isinstance(obj, collections.Iterable):
        return convert_list(obj, ids, parent, attr_type, item_func, cdata)
    raise TypeError('Unsupported data type: %s (%s)' % (obj, type(obj).__name__))


def convert_dict(obj, ids, parent, attr_type, item_func, cdata):
    """
    Converts a dict into an XML string
    """

    output = []
    addline = output.append

    for key, val in obj.items():
        attr = {} if not ids else {'id': '%s' % (get_unique_id(parent))}

        key, attr = make_valid_xml_name(key, attr)

        if isinstance(val, bool):
            addline(convert_bool(key, val, attr_type, attr))
        elif isinstance(val, str) or isinstance(val, numbers.Number):
            addline(convert_kv(key, val, attr_type, attr, cdata))
        elif hasattr(val, 'isoformat'):  # datetime
            addline(convert_kv(key, val.isoformat(), attr_type, attr, cdata))
        elif isinstance(val, dict):
            if attr_type:
                attr['type'] = get_xml_type(val)
            addline('<%s%s>%s</%s>' % (key,
                                       make_attrstring(attr),
                                       convert_dict(val, ids, key, attr_type, item_func, cdata),
                                       key))
        elif isinstance(val, collections.Iterable):
            if attr_type:
                attr['type'] = get_xml_type(val)
            addline('<%s%s>%s</%s>' % (key,
                                       make_attrstring(attr),
                                       convert_list(val, ids, key, attr_type, item_func, cdata),
                                       key))
        elif val is None:
            addline(convert_none(key, val, attr_type, attr))
        else:
            raise TypeError('Unsupported data type: %s (%s)' % (val, type(val).__name__))

    return ''.join(output)


def convert_list(items, ids, parent, attr_type, item_func, cdata):
    """
    Converts a list into an XML string
    """

    output = []
    addline = output.append

    item_name = item_func(parent)

    if ids:
        this_id = get_unique_id(parent)
    else:
        this_id = 0

    for i, item in enumerate(items):
        attr = {} if not ids else {'id': '%s_%s' % (this_id, i + 1)}
        if type(item) == bool:
            addline(convert_bool(item_name, item, attr_type, attr))
        elif isinstance(item, str) or isinstance(item, numbers.Number):
            addline(convert_kv(item_name, item, attr_type, attr, cdata))
        elif hasattr(item, 'isoformat'):  # datetime
            addline(convert_kv(item_name, item.isoformat(), attr_type, attr, cdata))
        elif isinstance(item, dict):
            if not attr_type:
                addline('<%s>%s</%s>' % (
                    item_name,
                    convert_dict(item, ids, parent, attr_type, item_func, cdata),
                    item_name,
                )
                        )
            else:
                addline('<%s type="dict">%s</%s>' % (
                    item_name,
                    convert_dict(item, ids, parent, attr_type, item_func, cdata),
                    item_name,
                )
                        )

        elif isinstance(item, collections.Iterable):
            if not attr_type:
                addline('<%s %s>%s</%s>' % (
                    item_name, make_attrstring(attr),
                    convert_list(item, ids, item_name, attr_type, item_func, cdata),
                    item_name,
                )
                        )
            else:
                addline('<%s type="list"%s>%s</%s>' % (
                    item_name, make_attrstring(attr),
                    convert_list(item, ids, item_name, attr_type, item_func, cdata),
                    item_name,
                )
                        )

        elif item is None:
            addline(convert_none(item_name, None, attr_type, attr))

        else:
            raise TypeError('Unsupported data type: %s (%s)' % (
                item, type(item).__name__)
                            )
    return ''.join(output)


def convert_kv(key, val, attr_type, attr=None, cdata=False):
    """
    Converts a number or string into an XML element
    """

    if attr is None:
        attr = {}
    key, attr = make_valid_xml_name(key, attr)

    if attr_type:
        attr['type'] = get_xml_type(val)
    attrstring = make_attrstring(attr)
    return '<%s%s>%s</%s>' % (
        key, attrstring,
        wrap_cdata(val) if cdata else escape_xml(val),
        key
    )


def convert_bool(key, val, attr_type, attr=None):
    """
    Converts a boolean into an XML element
    """

    if attr is None:
        attr = {}
    key, attr = make_valid_xml_name(key, attr)

    if attr_type:
        attr['type'] = get_xml_type(val)
    attrstring = make_attrstring(attr)
    return '<%s%s>%s</%s>' % (key, attrstring, str(val).lower(), key)


def convert_none(key, val, attr_type, attr=None):
    """
    Converts a null value into an XML element
    """

    if attr is None:
        attr = {}
    key, attr = make_valid_xml_name(key, attr)

    if attr_type:
        attr['type'] = get_xml_type(val)
    attrstring = make_attrstring(attr)
    return '<%s%s></%s>' % (key, attrstring, key)


def dicttoxml(obj, root=True, custom_root='root', ids=False, attr_type=True, item_func=default_item_func, cdata=False):
    """
    Converts a python object into XML
    :param obj:
    :param root: specifies whether the output is wrapped in an XML root element (bool, True by default)
    :param custom_root: specify a custom root element (str, 'root' by default)
    :param ids: specifies whether elements get unique ids (bool, False by default)
    :param attr_type: specifies whether elements get a data type attribute (bool, True by default)
    :param item_func: specifies what function should generate the element name for items in a list (function object)
    :param cdata: specifies whether string values should be wrapped in CDATA sections (bool, False by default)
    :return: (str)
    """

    output = []
    addline = output.append
    if root:
        addline('<?xml version="1.0" encoding="UTF-8" ?>')
        addline('<%s>%s</%s>' % (
            custom_root,
            convert(obj, ids, attr_type, item_func, cdata, parent=custom_root),
            custom_root,
        )
                )
    else:
        addline(convert(obj, ids, attr_type, item_func, cdata, parent=''))
    return ''.join(output).encode('utf-8')
