// SPDX-FileCopyrightText: 2022 Peter Urban, GEOMAR Helmholtz Centre for Ocean Research Kiel
// SPDX-FileCopyrightText: 2022 Sven Schorge, GEOMAR Helmholtz Centre for Ocean Research Kiel
// SPDX-FileCopyrightText: 2022 - 2025 Peter Urban, Ghent University
//
// SPDX-License-Identifier: MPL-2.0
//

#pragma once

/* generated doc strings */
#include ".docstrings/pingcontainer.doc.hpp"

/* std includes */
#include <filesystem>
#include <fstream>
#include <unordered_map>
#include <vector>

/* third party libraries */
#include <fmt/format.h>
#include <xtensor/containers/xtensor.hpp> 

// boost includes
#include <boost/sort/sort.hpp> // for sort

/* themachinethatgoesping includes */
#include <themachinethatgoesping/navigation/sensorconfiguration.hpp>
#include <themachinethatgoesping/tools/classhelper/objectprinter.hpp>
#include <themachinethatgoesping/tools/pyhelper/pyindexer.hpp>
#include <themachinethatgoesping/tools/timeconv.hpp>


namespace themachinethatgoesping {
namespace pingprocessing {
namespace datatypes {

template<typename t_Ping>
using PingVector = std::vector<std::shared_ptr<t_Ping>>;

template<typename t_Ping>
class PingContainer
{
  public:
    using type_Ping = t_Ping;

  protected:
    std::string _name;

    PingVector<type_Ping>      _pings;
    tools::pyhelper::PyIndexer _pyindexer;

  public:
    /**
     * @brief Construct a new empty PingContainer object
     *
     */
    PingContainer(std::string_view name = "PingContainer")
        : _name(name)
        , _pyindexer(0)
    {
    }

    /**
     * @brief Construct a new PingContainer object from a vector of pings
     *
     * @param pings: vector of pings
     */
    PingContainer(PingVector<type_Ping> pings, std::string_view name = "PingContainer")
        : _name(name)
        , _pings(std::move(pings))
        , _pyindexer(_pings.size())
    {
    }

    virtual ~PingContainer() = default;

    void add_ping(std::shared_ptr<type_Ping> ping)
    {
        this->_pings.push_back(std::move(ping));
        this->_pyindexer.reset(this->_pings.size());
    }

    void add_pings(const PingVector<type_Ping>& ping)
    {
        this->_pings.insert(this->_pings.end(), ping.begin(), ping.end());
        this->_pyindexer.reset(this->_pings.size());
    }

    void add_pings(PingVector<type_Ping>&& ping)
    {
        if (this->_pings.empty())
        {
            this->_pings = std::move(ping);
        }
        else
        {
            this->_pings.insert(this->_pings.end(),
                                std::make_move_iterator(ping.begin()),
                                std::make_move_iterator(ping.end()));
        }
        this->_pyindexer.reset(this->_pings.size());
    }

    /**
     * @brief Add a ping without resetting the pyindexer.
     * Call reindex() after all pings have been added.
     */
    void add_ping_no_reindex(std::shared_ptr<type_Ping> ping)
    {
        this->_pings.push_back(std::move(ping));
    }

    /**
     * @brief Reset the pyindexer after batch additions via add_ping_no_reindex.
     */
    void reindex()
    {
        this->_pyindexer.reset(this->_pings.size());
    }

    void set_pings(PingVector<type_Ping> pings)
    {
        _pings = std::move(pings);
        _pyindexer.reset(_pings.size());
    }

    const PingVector<type_Ping>& get_pings() const& { return _pings; }
    PingVector<type_Ping> get_pings() && { return std::move(_pings); }

    // ----- compute ping information -----
    size_t max_number_of_samples() const
    {
        size_t max_samples = 0;
        for (size_t i : _pyindexer)
        {
            if (!_pings[i]->has_watercolumn())
                continue;

            size_t local_max =
                xt::amax(_pings[i]->watercolumn().get_number_of_samples_per_beam())();
            max_samples = std::max(max_samples, local_max);
        }

        return max_samples;
    }

    // ----- iterator interface -----

    PingContainer<type_Ping> reversed() const
    {
        return this->operator()(_pyindexer.reversed().to_slice());
    }

    PingContainer<type_Ping> operator()(const tools::pyhelper::PyIndexer::Slice& slice) const
    {
        PingContainer<type_Ping> ping_container(*this);

        tools::pyhelper::PyIndexer pyindexer(_pings.size(), slice);

        PingVector<type_Ping> pings;
        pings.reserve(pyindexer.size());

        for (size_t i : pyindexer)
        {
            pings.push_back(_pings[i]);
        }

        ping_container.set_pings(std::move(pings));

        return ping_container;
    }

    PingContainer<type_Ping> operator()(const std::string& channel_id) const
    {
        PingContainer<type_Ping> filtered(*this);

        PingVector<type_Ping> pings;

        for (const auto& ping : _pings)
        {
            if (ping->get_channel_id() == channel_id)
                pings.push_back(ping);
        }

        filtered.set_pings(std::move(pings));

        return filtered;
    }

    PingContainer<type_Ping> operator()(const std::vector<std::string>& channel_ids) const
    {
        PingContainer<type_Ping> filtered(*this);

        PingVector<type_Ping> pings;

        for (const auto& ping : _pings)
        {
            if (std::find(channel_ids.begin(), channel_ids.end(), ping->get_channel_id()) !=
                channel_ids.end())
                pings.push_back(ping);
        }

        filtered.set_pings(std::move(pings));

        return filtered;
    }

    std::map<std::string, size_t> count_pings_per_channel_id() const
    {
        std::map<std::string, size_t> channel_ids;

        for (const auto& ping : _pings)
        {
            channel_ids[ping->get_channel_id()] += 1;
        }

        return channel_ids;
    }

    std::vector<std::string> find_channel_ids() const
    {
        std::set<std::string> channel_ids;

        for (const auto& ping : _pings)
        {
            channel_ids.insert(ping->get_channel_id());
        }

        std::vector<std::string> vec;
        std::copy(channel_ids.begin(), channel_ids.end(), std::back_inserter(vec));
        return vec;
    }

    /**
     * @brief  Split the data if the time difference between two subsequent datagrams is larger than
     * arg Note: for this function to make sense the data should be sorted_in_time
     *
     * @param max_time_diff_seconds: maximum time difference between two subsequent datagrams in
     * seconds
     * @return std::vector<PingContainer<type_Ping>>
     */
    std::vector<PingContainer<type_Ping>> split_by_time_diff(double max_time_diff_seconds) const
    {
        std::vector<PingContainer<type_Ping>> containers;

        PingVector<type_Ping> pings;

        for (const auto& ping : _pings)
        {
            if (!pings.empty())
            {
                if (ping->get_timestamp() - pings.back()->get_timestamp() > max_time_diff_seconds)
                {
                    containers.push_back(PingContainer<type_Ping>(pings));
                    pings = PingVector<type_Ping>();
                }
            }
            pings.push_back(ping);
        }

        containers.push_back(PingContainer<type_Ping>(pings));

        return containers;
    }

    // sort _datagram_infos_all by timestamp in _datagram_timestamps
    PingContainer<type_Ping> get_sorted_by_time() const
    {
        PingContainer<type_Ping> sorted(*this);
        // Your function
        auto& pings = sorted._pings;
        // sort _datagram_infos_all by  time, then file_pos then file number
        // TODO: this is faster than std sort, but python sorting (timsort?) seems
        // to be 10x faster for this use case
        boost::sort::pdqsort(pings.begin(), pings.end(), [](const auto& lhs, const auto& rhs) {
            if (lhs->get_timestamp() < rhs->get_timestamp())
                return true;
            return false;
        });

        return sorted;
    }

    size_t size() const { return _pyindexer.size(); }

    std::shared_ptr<type_Ping> at(int64_t index) const { return _pings[_pyindexer(index)]; }

    // ----- common info functions -----

    /**
     * @brief Compute some time statistics for the pings in the container
     * The is_sorted variable is interpreted as follows:
     *  - 1: the pings are sorted by time (ascending)
     *  - 0: the pings are not sorted by time
     * - -1: the pings are sorted by time (descending)
     *
     * @return std::tuple<min_timestamp, max_timestamp, is_sorted()>
     */
    std::tuple<double, double, int> timeinfo() const
    {
        double min_time  = std::numeric_limits<double>::max();
        double max_time  = std::numeric_limits<double>::min();
        int    is_sorted = 1;

        for (size_t i : _pyindexer)
        {
            const auto& ping = _pings[i];
            min_time         = std::min(min_time, ping->get_timestamp());
            max_time         = std::max(max_time, ping->get_timestamp());

            if (i > 1)
            {
                if (is_sorted == 1)
                {
                    if (ping->get_timestamp() < _pings[i - 1]->get_timestamp())
                        is_sorted = 0;
                }
                else if (is_sorted == -1)
                {
                    if (ping->get_timestamp() > _pings[i - 1]->get_timestamp())
                        is_sorted = 0;
                }
            }
            else if (i == 1)
            {
                if (ping->get_timestamp() < _pings[i - 1]->get_timestamp())
                    is_sorted = -1;
            }
        }

        return std::make_tuple(min_time, max_time, is_sorted);
    }

    /**
     * @brief  Split the data into a map of sensor configurations
     * TODL: this function makes pybind11_mkdoc crash
     *
     * @return ping container map
     */
    std::unordered_map<navigation::SensorConfiguration, PingContainer<type_Ping>>
    split_by_sensor_configuration() const
    {
        std::unordered_map<navigation::SensorConfiguration, PingContainer<type_Ping>> containers;

        for (const auto& ping : _pings)
        {
            auto sensor_configuration = ping->get_sensor_configuration();
            containers[sensor_configuration].add_ping(ping);
        }

        return containers;
    }

    // ----- objectprinter -----
    tools::classhelper::ObjectPrinter __printer__(unsigned int float_precision,
                                                  bool         superscript_exponents) const
    {
        tools::classhelper::ObjectPrinter printer(_name, float_precision, superscript_exponents);

        printer.register_section("Time info");
        double min_time, max_time;
        int    is_sorted;
        std::tie(min_time, max_time, is_sorted) = timeinfo();
        std::string is_sorted_str               = is_sorted == 1    ? "ascending"
                                                  : is_sorted == -1 ? "descending"
                                                                    : "no";
        std::string time_str_min =
            tools::timeconv::unixtime_to_datestring(min_time, 2, "%d/%m/%Y %H:%M:%S");
        std::string time_str_max =
            tools::timeconv::unixtime_to_datestring(max_time, 2, "%d/%m/%Y %H:%M:%S");
        printer.register_value("Start time", time_str_min);
        printer.register_value("End time", time_str_max);
        printer.register_value("Sorted", is_sorted_str);

        printer.register_section("Contained pings");
        auto channel_ids = find_channel_ids();

        if (channel_ids.size() > 1)
            printer.register_value("Total", size());

        for (const auto& channel_id : channel_ids)
            printer.register_value(channel_id, this->operator()(channel_id).size());

        return printer;
    }

    // -- class helper function macros --
    // define info_string and print functions (needs the __printer__ function)
    __CLASSHELPER_DEFAULT_PRINTING_FUNCTIONS__
};

}
} // namespace pingprocessing
} // namespace themachinethatgoesping