from harvesters.core import Harvester

h = Harvester()
h.add_cti_file("/opt/spinnaker/lib/spinnaker-gentl/Spinnaker_GenTL.cti")
h.update_device_info_list()

print(h.cti_files)
print(h.device_info_list)
