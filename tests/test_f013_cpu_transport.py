import hashlib
import stat
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from f013_cpu_qualifier import extract


class Transport(unittest.TestCase):
    def write(self, root, name, body, mode=stat.S_IFREG):
        archive=root/'fake.bin'
        with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_STORED) as z:
            info=zipfile.ZipInfo(name)
            info.external_attr=(mode|0o600)<<16
            z.writestr(info,body)
        inventory={name:{'bytes':len(body),'sha256':hashlib.sha256(body).hexdigest()}}
        return archive,inventory

    def test_exact_positive_inventory_roundtrip_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            archive,inventory=self.write(root,'source/a.py',b'# artificial only\n')
            destination=root/'fresh'
            extract(archive,inventory,destination,exact=True)
            self.assertEqual((destination/'source/a.py').read_bytes(),b'# artificial only\n')
            with self.assertRaises(ValueError):
                extract(archive,inventory,destination,exact=True)

    def test_nonfinite_source_hash_or_extra_inventory_refused_before_directory(self):
        for fault in ('hash','size','foreign'):
            with tempfile.TemporaryDirectory() as temp:
                root=Path(temp)
                archive,inventory=self.write(root,'source/a.py',b'# fake\n')
                if fault=='hash':
                    inventory['source/a.py']['sha256']='0'*64
                elif fault=='size':
                    inventory['source/a.py']['bytes']+=1
                else:
                    inventory['foreign.py']=inventory['source/a.py']
                with self.assertRaises(ValueError):
                    extract(archive,inventory,root/'fresh',exact=True)
                self.assertFalse((root/'fresh').exists())

    def test_path_escape_and_symlink_refused(self):
        for name,mode in [('../escape',stat.S_IFREG),('/absolute',stat.S_IFREG),('safe/link',stat.S_IFLNK)]:
            with tempfile.TemporaryDirectory() as temp:
                root=Path(temp)
                archive,inventory=self.write(root,name,b'fake',mode)
                with self.assertRaises(ValueError):
                    extract(archive,inventory,root/'fresh',exact=True)
                self.assertFalse((root/'fresh').exists())

    def test_generic_subset_extracts_only_selected_sources(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            archive,inventory=self.write(root,'generic/fake.py',b'# generic\n')
            with zipfile.ZipFile(archive,'a') as z:
                z.writestr('trained/unused.pt',b'not opened')
            extract(archive,inventory,root/'fresh',exact=False)
            self.assertTrue((root/'fresh/generic/fake.py').is_file())
            self.assertFalse((root/'fresh/trained').exists())


if __name__=='__main__':
    unittest.main()
