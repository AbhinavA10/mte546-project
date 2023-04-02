# Export Set of longitude/latitude to kml file for plotting
COORD_LIST = """
            -112.2550785337791,36.07954952145647
            -112.2549277039738,36.08117083492122
            -112.2552505069063,36.08260761307279
            -112.2564540158376,36.08395660588506
            -112.2580238976449,36.08511401044813
            -112.2595218489022,36.08584355239394
            -112.2608216347552,36.08612634548589
            -112.262073428656,36.08626019085147
            -112.2633204928495,36.08621519860091
            -112.2644963846444,36.08627897945274
            -112.2656969554589,36.08649599090644 
"""

from lxml import etree

root = etree.parse('template.kml').getroot()

tags = root.findall('.//coordinates', {None : 'http://www.opengis.net/kml/2.2'}) # recurisvely find all coordinate tags in namespace
ground_truth_tag = tags[0]
estimation_tag = tags[1]
ground_truth_tag.text = COORD_LIST
estimation_tag.text = COORD_LIST

with open('output.kml', 'wb') as f:
    f.write(etree.tostring(root, xml_declaration=True, encoding='UTF-8', pretty_print=True))
