class classproperty(property):
    def __get__(self, cls, owner):
        return classmethod(self.fget).__get__(None, owner)()

class Singleton:
    _instance = None

    def __init__(self):
        self.data = "Default Data"

    @classproperty
    def instance(cls):
        """
        Class-level property to access the singleton instance.
        """
        if cls._instance is None:
            cls._instance = Singleton()  # Create the singleton instance
        return cls._instance


# Usage
singleton1 = Singleton.instance  # Access the singleton instance
print(singleton1.data)           # Output: Default Data

singleton1.data = "Updated Data"  # Modify an attribute

singleton2 = Singleton.instance   # Access the same singleton instance
print(singleton2.data)            # Output: Updated Data

# Verify that both variables point to the same instance
print(singleton1 is singleton2)   # Output: True
