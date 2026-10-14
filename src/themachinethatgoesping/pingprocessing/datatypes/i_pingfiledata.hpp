// SPDX-FileCopyrightText: 2022 Peter Urban, GEOMAR Helmholtz Centre for Ocean Research Kiel
// SPDX-FileCopyrightText: 2022 Sven Schorge, GEOMAR Helmholtz Centre for Ocean Research Kiel
// SPDX-FileCopyrightText: 2022 - 2025 Peter Urban, Ghent University
//
// SPDX-License-Identifier: MPL-2.0
//

#pragma once

/* generated doc strings */
#include ".docstrings/i_pingfiledata.doc.hpp"

/* std includes */
#include <filesystem>
#include <fstream>
#include <set>
#include <unordered_map>
#include <vector>

#include <fmt/format.h>

// xtensor includes
#include <xtensor/containers/xadapt.hpp>


#include <xtensor/views/xview.hpp>

/* themachinethatgoesping includes */
#include <themachinethatgoesping/navigation/navigationinterpolatorlatlon.hpp>
#include <themachinethatgoesping/tools/classhelper/objectprinter.hpp>




#include <themachinethatgoesping/algorithms/pingtools/beamsampleselection.hpp>

namespace themachinethatgoesping {
namespace pingprocessing {
namespace datatypes {

/**
 * @class I_PingFileData
 * @brief Interface for raw ping data.
 *
 * This class defines an interface for raw ping data. It provides methods to access and manipulate
 * the properties of the ping data, such as the name, file ping counter, primary file number, file numbers,
 * primary file path, and file paths.
 *
 * The class also includes a nested exception class, `not_implemented`, which is thrown when a method is not implemented.
 *
 * The class provides a `__printer__` function for object printing, which can be used to print the properties
 * of the ping data.
 */
class I_PingFileData {
    size_t _file_ping_counter = 0; ///< Counter of the ping in the file (starting at 0)
    size_t _primary_file_nr = 0;   ///< File number of the primary file

protected:
    virtual std::string class_name() const { return "I_PingFileData"; }

public:
    /**
     * @brief Constructor for I_PingFileData.
     * @param name The name of the ping data.
     */
    I_PingFileData() = default;

    /**
     * @brief Destructor for I_PingFileData.
     */
    virtual ~I_PingFileData() = default;

    /**
     * @brief Get the file ping counter.
     * @return size_t The file ping counter.
     */
    size_t get_file_ping_counter() const { return _file_ping_counter; }

    /**
     * @brief Set the file ping counter.
     * @param file_ping_counter The file ping counter to set.
     */
    void set_file_ping_counter(size_t file_ping_counter) { _file_ping_counter = file_ping_counter; }

    /**
     * @brief Set the primary file number for this ping.
     * @param primary_file_nr The primary file number to set.
     */
    void set_primary_file_nr(size_t primary_file_nr) { _primary_file_nr = primary_file_nr; }

    /**
     * @brief Get the primary file number of this ping.
     * @return size_t The primary file number.
     */
    size_t get_primary_file_nr() const { return _primary_file_nr; }

    /**
     * @brief Get the file numbers of the contained datagrams.
     * @return std::vector<size_t> The file numbers.
     * @throws not_implemented Exception if not implemented.
     */
    virtual std::vector<size_t> get_file_numbers() const {
        throw not_implemented("get_file_numbers", this->class_name());
    }

    /**
     * @brief Get the primary file path of this ping.
     * @return std::string The primary file path of this ping.
     * @throws not_implemented Exception if not implemented.
     */
    virtual std::string get_primary_file_path() const {
        throw not_implemented("get_primary_file_path", this->class_name());
    }

    /**
     * @brief Get the file paths.
     * @return std::vector<std::string> The file paths associated with the contained datagrams.
     * @throws not_implemented Exception if not implemented.
     */
    virtual std::vector<std::string> get_file_paths() const {
        throw not_implemented("get_file_paths", this->class_name());
    }

protected:
    /**
     * @struct not_implemented
     * @brief Exception thrown when a method is not implemented.
     */
    struct not_implemented : public std::runtime_error {
        /**
         * @brief Constructor for not_implemented.
         * @param method_name The name of the method.
         * @param name The name of the ping data.
         */
        not_implemented(std::string_view method_name, std::string_view name)
                : std::runtime_error(
                            fmt::format("raw data method {} not implemented for ping type '{}'", method_name, name)) {}
    };

public:
    /**
     * @brief ObjectPrinter for I_PingFileData.
     * @param float_precision The float precision for printing.
     * @return tools::classhelper::ObjectPrinter The ObjectPrinter instance.
     */
    tools::classhelper::ObjectPrinter __printer__(unsigned int float_precision, bool superscript_exponents) const;

    // -- class helper function macros --
    // define info_string and print functions (needs the __printer__ function)
    __CLASSHELPER_DEFAULT_PRINTING_FUNCTIONS__
};

}
} // namespace pingprocessing
} // namespace themachinethatgoesping