from ipywidgets import interact

def greeting(text="World"):
     print("Hello {}".format(text))

def i(func):
  return interact(func, text="IPython Widgets")