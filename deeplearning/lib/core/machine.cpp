
#include <liblearning/core/machine.h>

using namespace core;

void machine::setMachineID(const string& machineID_)
{
  machineID = machineID_;
}
const string & machine::getMachineID()
{
  return machineID;
}
