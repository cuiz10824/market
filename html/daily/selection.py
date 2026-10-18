"""Shared eight-index scope for training, model loading and daily collection."""
import json


def load_selection(root):
    profile=json.loads((root/'data/active_indices.json').read_text(encoding='utf-8'))
    indices=profile['indices']
    codes=[item['code'] for item in indices]
    if profile.get('schema_version')!=1 or len(codes)!=8 or len(set(codes))!=8:
        raise ValueError('当前指数配置必须包含8个互不重复的指数')
    registry={item['code']:item for item in json.loads((root/'data/registry.json').read_text(encoding='utf-8')) if item.get('secid')}
    if any(code not in registry for code in codes):raise ValueError('指数不在已核验主数据中')
    items=[]
    for item in indices:
        meta=dict(registry[item['code']])
        meta.update(display_name=item['display_name'],selection_note=item['rationale'])
        items.append(meta)
    return items


def scoped_protocol(root,protocol):
    result=dict(protocol)
    codes=[item['code'] for item in load_selection(root)]
    if not set(codes)<=set(protocol['codes']):raise ValueError('训练协议未覆盖全部当前指数')
    result['codes']=codes
    result['scope']='data/active_indices.json'
    return result
