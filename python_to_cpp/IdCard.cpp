#include "IdCard.h"

std::string IdCard::get_name()
{
    return this->name;
}

std::string IdCard::get_dob()
{
    return this->dob;
}

std::string IdCard::get_addr()
{
    return this->addr;
}

void IdCard::set_name(std::string name)
{
    this->name = name;
}

void IdCard::set_addr(std::string addr)
{
    this->addr = addr;
}

void IdCard::set_dob(std::string dob)
{
    this->dob = dob;
}

IdCard::IdCard()
{
    this->name = "unnamed";
    this->dob = "dd-mm-yyyy";
    this->addr = "default_address";
}

IdCard::IdCard(std::string name, std::string addr, std::string dob)
{
    this->name = name;
    this->dob = dob;
    this->addr = addr;
}

IdCard::~IdCard()
{
   std::cout << "Object is being deleted" << std::endl;
}
