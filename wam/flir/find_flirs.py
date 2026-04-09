from harvesters.core import Harvester

h = Harvester()
h.add_file("/opt/spinnaker/lib/spinnaker-gentl/Spinnaker_GenTL.cti")
h.update()
print(h.files)
print(h.device_info_list)