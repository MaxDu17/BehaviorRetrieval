

class Task:
    """Interface for subtask definition."""
    REWARD = 1.0        # reward that's received upon completion

    def done(self, info):
        raise NotImplementedError("Task classes need to define their success condition.")


class PickPlaceTask(Task):
    def __init__(self, object, target_object, pos, target_pos):
        self._object = object
        self._target_object = target_object
        self._pos = pos
        self._target_pos = target_pos

    def done(self, info):
        return info['place_success']

    @property
    def object(self):
        return self._object

    @property
    def target_pos(self):
        return self._target_pos


class PickTask(Task):
    def __init__(self, object, target_object, pos, target_pos):
        self._object = object
        self._target_object = target_object
        self._pos = pos
        self._target_pos = target_pos

    def done(self, info):
        return info['grasp_success']

    @property
    def object(self):
        return self._object

class PlaceTask(Task):
    def __init__(self, object, target_object, pos, target_pos):
        self._object = object
        self._target_object = target_object
        self._pos = pos
        self._target_pos = target_pos

    def done(self, info):
        return info['place_success']

    @property
    def object(self):
        return self._object

    @property
    def target_pos(self):
        return self._target_pos

class DrawerOpenTask(Task):   
    def done(self, info):
        return info['drawer_opened']


class DrawerClosedTask(Task):
    def done(self, info):
        return info['drawer_closed']

