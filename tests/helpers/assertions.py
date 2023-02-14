from approval_utilities.utilities.exceptions.multiple_exceptions import MultipleExceptions


def gather_all_exceptions(params, code_to_execute):
    class _Collector:
        def __init__(self):
            self.exceptions = []

        def add(self, exception):
            self.exceptions.append(exception)

        def assert_any_is_true(self):
            if len(params) == len(self.exceptions):
                raise MultipleExceptions(self.exceptions)

    collector = _Collector()
    for p in params:
        try:
            code_to_execute(p)
        except Exception as e:
            collector.add(e)

    return collector
