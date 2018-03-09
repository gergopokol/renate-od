from lxml import etree
import os


class Settings:
    """
    This class is to read and write setup information stored in XML files.
    """
    def __init__(self,
                 filename='setup.xml'):

        self.filename = filename
        self.tree = 'File content'
        self.dict = {'Filename' : self.filename}
        try:
            self.tree = etree.parse(filename)
            print('Settings read from ' + filename)
        except:
            print('Could not read settings read from ' + filename)
            self.new_tree()
            self.write_to_file()
        self.parse_to_dictionary()

    def new_tree(self):
        """
        This creates new tree structure with basic parts.
        """
        page = etree.Element('xml')
        self.tree = etree.ElementTree(page)
        head_element = etree.SubElement(page, 'head')
        body_element = etree.SubElement(page, 'body')
        title = etree.SubElement(head_element, 'title')
        title.text = os.path.splitext(os.path.split(self.filename)[1])[0]  # Create title from file name.

    def write_to_file(self):
        """
        This writes tree structure to the corresponding XML file.
        """
        self.tree.write(self.filename, pretty_print=True)
        print('Setting file written to ' + self.filename)

    def append_element(self, parent_element='', new_element='test_element', new_text='Test element text'):
        """
        This appends a new element to the parent element.
        :param parent_element: Parent etree.Element
        :param new_element: Name of new child
        :param new_text: Text content of new child
        """
        if parent_element == '':
            parent_element = self.tree.find('body')  # Set body as default parent element.
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
            element = self.tree.find('body')  # Set body as default element.
        element.set(attribute, value)

    def return_attribute(self, element_name='test_element', attribute_name='test_attribute'):
        """
        This sets an attribute of an element to a value.
        :param element_name: Element to read the attribute of
        :param attribute_name: Attribute to read
        :return: Value of attribute
        """
        element = self.tree.find(element_name)
        return element.attribute

    def return_text(self, parent_element='', element_name='test_element'):
        """
        This returns text content of an element with element_name
        :param parent_element: Parent etree.Element
        :param element_name: Unique name of element
        :return: Text content of the element
        """
        if parent_element == '':
            parent_element = self.tree.find('body')  # Set body as default parent element.
        element = parent_element.find(element_name)
        return element.text

    def parse_to_dictionary(self):
        """
        This parses all elements in the 'body' into a dictionary. Attributes not supported.
        """
        parent_element = self.tree.find('body')
        for element in parent_element.getchildren():
            self.dict[element.tag] = element.text