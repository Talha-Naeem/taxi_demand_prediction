import hopsworks

def login_to_hopsworks():
    """
    Login to Hopsworks project and return project + feature store
    """
    project = hopsworks.login()
    fs = project.get_feature_store()
    return project, fs
