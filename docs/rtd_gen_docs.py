import os
from os.path import abspath, join, dirname
from shutil import copyfile

curdir = abspath(dirname(__file__))
rootdir = dirname(curdir)
conf_dir = join(rootdir, 'conf')
apidoc_cmd = 'sphinx-apidoc'
apidoc_exclusions = join(rootdir, 'pyMS', 'test')
apidoc_params = '-f -o {} {} {}'.format(curdir, rootdir, apidoc_exclusions)


def main():
    copyfile(join(rootdir, 'README.md'), join(curdir, 'README.md'))
    os.system('{} {}'.format(apidoc_cmd, apidoc_params))


if __name__ == '__main__':
    main()
