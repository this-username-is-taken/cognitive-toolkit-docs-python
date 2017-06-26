# -*- coding: utf-8 -*-

import io
import os
import yaml

# Write YAML file
def rewrite_yml(data, yml_file):
    with io.open(yml_file, 'w', encoding='utf8') as outfile:
        yaml.dump(data, outfile, default_flow_style=False, allow_unicode=True)


toc_yml_full_path = os.getcwd() + '\\_build\\docfx_yaml\\toc.yml'

with open(toc_yml_full_path, 'r') as stream:
    try:
        data_loaded = yaml.load(stream)

        # should only have one root node: cntk
        if len(data_loaded) == 1:
            cntk_node = data_loaded[0]
            if 'name' in cntk_node:
                print(cntk_node['name'])
                cntk_node['name'] = 'Reference'

            # change leave 2 node's name: remove 'cntk' prefix
            if 'items' in cntk_node:
                for item in cntk_node['items']:
                    if 'name' in item:
                        if item['name'].startswith('cntk.'):
                            item['name'] = item['name'][5:]
                            print('update old name to %s' % item['name'])

        rewrite_yml(data_loaded, toc_yml_full_path)
    except yaml.YAMLError as exc:
        print(exc)