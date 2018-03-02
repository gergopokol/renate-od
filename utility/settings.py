from lxml import etree
import os


class Settings:
    """
    This class is to read and write setup information stored in XML files.
    """
    def __init__(self,
                 filename='setup.xml'):

        self.filename = filename
        self.doc = 'File content'
        try:
            self.doc = etree.parse(filename)
            print('Settings read from ' + filename)
        except:
            print('Could not read settings read from ' + filename)
            self.new_doc()
            self.write_to_file()

    def new_doc(self):
        """
        This creates new doc structure with basic parts.
        """
        page = etree.Element('html')
        self.doc = etree.ElementTree(page)
        head_element = etree.SubElement(page, 'head')
        body_element = etree.SubElement(page, 'body')
        title = etree.SubElement(head_element, 'title')
        title.text = os.path.splitext(os.path.split(self.filename)[1])[0]  # Create title from file name.

    def write_to_file(self):
        """
        This writes doc structure to the corresponding XML file.
        """
        self.doc.write(self.filename, pretty_print=True)
        print('Setting file written to ' + self.filename)

    def append_element(self, parent_element='', new_element='test_element', new_text='Test element text'):
        """
        This appends a new element to the parent element.
        :param parent_element: Parent etree.Element
        :param new_element: Name of new child
        :param new_text: Text content of new child
        """
        if parent_element == '':
            parent_element = self.doc.find('body')  # Set body as default parent element.
        new_element = etree.Element(new_element)
        new_element.text = new_text
        parent_element.append(new_element)

    def set_attribute(self, element='', attribute='test_attribute', value='Test value'):
        """
        This sets an attribute of an element to a value.
        :param element: Element to set the attribute of
        :param attribute: Attribute to set
        :param value: Value to set
        """
        if element == '':
            element = self.doc.find('body')  # Set body as default element.
        element.set(attribute, value)

    def return_text(self, parent_element='', element_name='test_element'):
        """
        This returns text content of an element with element_name
        :param parent_element: Parent etree.Element
        :param element_name: Unique name of element
        :return: Text content of the element
        """
        if parent_element == '':
            parent_element = self.doc.find('body')  # Set body as default parent element.
        element = parent_element.find(element_name)
        return element.text

