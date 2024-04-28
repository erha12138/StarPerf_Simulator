class Parent:
    def __init__(self, parent_attr):
        self.parent_attr = parent_attr

class Child(Parent):
    def __init__(self, parent_attr, child_attr):
        super().__init__(parent_attr)  # 显式调用父类的__init__方法
        self.child_attr = child_attr
        print(self.parent_attr)

class Child2(Parent):
    def __init__(self, child_attr):
        super().__init__()  # 显式调用父类的__init__方法
        self.child_attr = child_attr
        print(self.parent_attr)

Child = Child(6,3)
Child2 = Child2(1)