// SPDX-FileCopyrightText: 2022 - 2025 Peter Urban, Ghent University
//
// SPDX-License-Identifier: MPL-2.0

#include "i_pingfiledata.hpp"

namespace themachinethatgoesping {
namespace pingprocessing {
namespace datatypes {

tools::classhelper::ObjectPrinter I_PingFileData::__printer__(unsigned int float_precision, bool superscript_exponents) const
{
    tools::classhelper::ObjectPrinter printer(this->class_name(), float_precision, superscript_exponents);

    printer.register_string("Primary source file", this->get_primary_file_path(),
                                                    std::to_string(this->get_primary_file_nr()));
    if (this->get_file_numbers().size() > 1) {
        auto file_numbers = this->get_file_numbers();
        auto file_paths = this->get_file_paths();
        for (size_t i = 1; i < file_numbers.size(); i++) {
            printer.register_string("Secondary source file", file_paths[i], std::to_string(file_numbers[i]));
        }
    }

    return printer;
}

} // namespace datatypes
} // namespace pingprocessing
} // namespace themachinethatgoesping
