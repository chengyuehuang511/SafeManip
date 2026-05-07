from typing import Union, Callable, Any, List


class Entity:
    pass


ID_ATTR = 'ENTITY_ID'


class SymbolicEntity(Entity):
    def __init__(self, name, base_filter: Union[Callable[[Any], bool], List[str]]):
        self.name = name
        self.base_filter = base_filter

    def is_valid(self, node):
        if type(self.base_filter) == list:
            return node.base_class in self.base_filter
        return self.base_filter(node)

    def __repr__(self):
        return self.name
    
    def __eq__(self, other):
        return self.name == other.name
    
    def __hash__(self):
        return hash(self.name)


class ConcreteEntity(Entity):
    def __init__(self, sym: SymbolicEntity, entity_id):
        self.sym = sym
        self.entity_id = entity_id

    def get_node(self, sg) -> Union[set, "UnboundEntityError"]:
        # TODO find way to cache this
        for node in sg.nodes:
            # some entities store their unique ID as their name
            if node.get_id() == self.entity_id:
                # create set with one element
                return {node}
        # the node was not found in the graph, return the empty set
        return set()

    def __repr__(self):
        return f'{self.sym.name}_{self.entity_id}'

    def get_node_name(self, sg):
        node_set = self.get_node(sg)
        if len(node_set) == 0:
            return None
        return next(iter(node_set)).name


class UnboundEntityError(Exception):
    def __init__(self, entities: List[SymbolicEntity]):
        self.entities = list(set(entities))
        super().__init__(f'No entities bound for {[entity.name for entity in self.entities]}')